import json
from pathlib import Path
from types import SimpleNamespace

from bop_toolkit_lib import inout
import cv2
import numpy as np
import pandas as pd
import torch
import trimesh

from aic_perception.utils.alignpose import nms_3d_base
from aic_perception.utils.alignpose import renderer_builder
from aic_perception.utils.alignpose.featuremetric_refiner_wrapper import refine_multiview_wrapper
from aic_perception.utils.alignpose.query_template_repre import QueryRepre, TemplateRepre
from aic_perception.utils.alignpose.renderer_base import RenderType
from aic_perception.utils.foundpose import feature_util, misc, pnp_util, projector_util, repre_util, structs

class PoseEstimator:
    def __init__(self, cameras, templates_dir, models_dir=None):
        self.cameras = cameras
        self.templates_dir = Path(templates_dir)
        self.models_dir = Path(models_dir) if models_dir is not None else self.templates_dir.parent / "ic" / "models"

        self.extractor_name = "dinov2_version=vits14-reg_stride=14_facet=token_layer=9_logbin=0_norm=1"
        self.device = "cuda"
        self.crop = True
        self.crop_rel_pad = 0.2
        self.crop_size = (420, 420)
        self.grid_cell_size = 14.0
        self.match_top_n_templates = 5
        self.match_top_k_buddies = 300
        self.pnp_ransac_iter = 400
        self.pnp_inlier_thresh = 10.0
        self.alignpose_num_iters = 30
        self.alignpose_loss_fn = "scaled_barron(-5, 0.5)"
        self.alignpose_nms_threshold = 0.1
        self.alignpose_nms_box_type = "axis-aligned"
        self.alignpose_nms_mode = "inter-object"

        self.extractor = feature_util.make_feature_extractor(self.extractor_name)
        self.extractor.to(self.device)
        self.extractor.eval()
        self.renderer = renderer_builder.build(
            renderer_type=renderer_builder.RendererType.PYRENDER_RASTERIZER,
            model_path=None,
            device=self.device,
        )

        self.template_dirs = {}
        for metadata_path in self.templates_dir.glob("*/metadata.json"):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.template_dirs[metadata[0]["lid"]] = metadata_path.parent

        self.objects = {}
        for object_id, template_dir in self.template_dirs.items():
            self.objects[object_id] = self.load_or_build_representation(object_id, template_dir)
            self.renderer.add_object_model(object_id, self.objects[object_id]["model_path"])
            print(f"Object {object_id}: representation ready", flush=True)

        self.camera_models = {}
        for camera_name, camera in self.cameras.items():
            intrinsics = camera["intrinsics"]
            extrinsics = camera["extrinsics"]
            T_w2c = np.eye(4)
            T_w2c[:3, :3] = np.array(extrinsics["R_w2c"]).reshape(3, 3)
            T_w2c[:3, 3] = np.array(extrinsics["t_w2c"]).reshape(3)
            self.camera_models[camera_name] = structs.PinholePlaneCameraModel(
                width=intrinsics["width"],
                height=intrinsics["height"],
                f=(intrinsics["fx"], intrinsics["fy"]),
                c=(intrinsics["cx"], intrinsics["cy"]),
                T_world_from_eye=np.linalg.inv(T_w2c),
            )

    def estimate_pose(self, camera_inputs):
        results = {}
        for camera_name, camera_input in camera_inputs.items():
            image = camera_input["color"]
            camera = self.camera_models[camera_name]
            results[camera_name] = {}
            for object_id, masks in camera_input["masks"].items():
                results[camera_name][object_id] = []
                for mask in masks:
                    pose = self.foundpose_inference(image, mask, camera, object_id)
                    results[camera_name][object_id].append(pose)
        return self.alignpose_inference(camera_inputs, results)

    def load_or_build_representation(self, object_id, template_dir):
        from aic_perception.utils.foundpose import knn_util

        repre_dir = template_dir / "repre"
        if not (repre_dir / "repre.pth").exists():
            print(f"Building representation for object {object_id} in {repre_dir}", flush=True)
            self.build_representation(object_id, template_dir, repre_dir)
        else:
            print(f"Loading representation for object {object_id} from {repre_dir}", flush=True)

        repre = repre_util.load_object_repre(str(repre_dir), tensor_device=self.device)
        for name, value in list(repre.__dict__.items()):
            if name in (
                "vertices",
                "feat_vectors",
                "feat_to_template_ids",
                "feat_cluster_centroids",
                "feat_cluster_idfs",
                "template_descs",
            ):
                setattr(repre, name, value.to(self.device))

        print(f"Building visual-word KNN index for object {object_id}", flush=True)
        visual_words_index = knn_util.KNN(
            k=repre.template_desc_opts.tfidf_knn_k,
            metric=repre.template_desc_opts.tfidf_knn_metric,
        )
        visual_words_index.fit(repre.feat_cluster_centroids)

        vertices_np = misc.tensor_to_array(repre.vertices)
        vertices_np = vertices_np[::max(1, len(vertices_np) // 5000)]

        return {
            "repre": repre,
            "visual_words_index": visual_words_index,
            "template_indices": {},
            "vertices_np": vertices_np,
            "model_path": str(self.models_dir / f"obj_{object_id:06d}.ply"),
            "mesh": trimesh.load_mesh(str(self.models_dir / f"obj_{object_id:06d}.ply")),
        }

    def build_representation(self, object_id, template_dir, repre_dir):
        from utils.foundpose import cluster_util, template_util

        with open(template_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        print(f"Object {object_id}: extracting DINO features from {len(metadata)} templates", flush=True)

        all_feats = []
        all_vertex_ids = []
        all_vertices = []
        all_template_ids = []
        all_templates = []
        all_template_cameras = []

        with torch.inference_mode():
            for i, sample in enumerate(metadata):
                template_id = sample["template_id"]
                if i % 10 == 0:
                    print(f"Object {object_id}: template {i + 1}/{len(metadata)}", flush=True)
                camera_data = sample["cameras"]
                camera = structs.PinholePlaneCameraModel(
                    width=camera_data["ImageSizeX"],
                    height=camera_data["ImageSizeY"],
                    f=(camera_data["fx"], camera_data["fy"]),
                    c=(camera_data["cx"], camera_data["cy"]),
                    T_world_from_eye=np.array(camera_data["T_WorldFromCamera"]),
                )

                rgb_path = template_dir / "rgb" / f"template_{template_id:04d}.png"
                depth_path = template_dir / "depth" / f"template_{template_id:04d}.png"
                mask_path = template_dir / "mask" / f"template_{template_id:04d}.png"

                image = inout.load_im(str(rgb_path))
                depth = inout.load_depth(str(depth_path))
                mask = inout.load_im(str(mask_path))

                image_chw = (
                    misc.array_to_tensor(image)
                    .to(torch.float32)
                    .permute(2, 0, 1)
                    .to(self.device)
                    / 255.0
                )
                depth_hw = misc.array_to_tensor(depth).to(torch.float32).to(self.device)
                mask_hw = misc.array_to_tensor(mask).to(torch.float32).to(self.device)

                T_world_from_model = np.eye(4)
                T_world_from_model[:3, :3] = np.array(sample["pose"]["R"])
                T_world_from_model[:3, 3:] = np.array(sample["pose"]["t"]).reshape(3, 1)
                T_model_from_camera = np.linalg.inv(T_world_from_model).dot(camera.T_world_from_eye)
                T_model_from_camera = (
                    misc.array_to_tensor(T_model_from_camera)
                    .to(torch.float32)
                    .to(self.device)
                )

                feats, vertex_ids, vertices = feature_util.get_visual_features_registered_in_3d(
                    image_chw=image_chw,
                    depth_image_hw=depth_hw,
                    object_mask=mask_hw,
                    camera=camera,
                    T_model_from_camera=T_model_from_camera,
                    extractor=self.extractor,
                    grid_cell_size=self.grid_cell_size,
                )

                all_feats.append(feats)
                all_vertex_ids.append(vertex_ids)
                all_vertices.append(vertices)
                all_template_ids.append(
                    template_id * torch.ones(feats.shape[0], dtype=torch.int32, device=self.device)
                )
                all_templates.append((image_chw * 255).to(torch.uint8))
                all_template_cameras.append(
                    camera.copy(
                        T_world_from_eye=np.linalg.inv(
                            misc.tensor_to_array(T_model_from_camera)
                        )
                    )
                )

        repre = repre_util.FeatureBasedObjectRepre(
            vertices=torch.cat(all_vertices),
            feat_vectors=torch.cat(all_feats),
            feat_opts=repre_util.FeatureOpts(extractor_name=self.extractor_name),
            feat_to_vertex_ids=torch.cat(all_vertex_ids),
            feat_to_template_ids=torch.cat(all_template_ids),
            templates=torch.stack(all_templates),
            template_cameras_cam_from_model=all_template_cameras,
        )

        print(f"Object {object_id}: fitting PCA", flush=True)
        pca = projector_util.PCAProjector(n_components=256, whiten=False)
        pca.fit(repre.feat_vectors, max_samples=100000)
        repre.feat_raw_projectors.append(pca)
        feat_vectors = pca.transform(repre.feat_vectors).to(torch.float32).contiguous()

        print(f"Object {object_id}: clustering features", flush=True)
        centroids, cluster_ids, _ = cluster_util.kmeans(
            samples=feat_vectors,
            num_centroids=2048,
            verbose=True,
        )
        repre.feat_vectors = feat_vectors
        repre.feat_cluster_centroids = centroids
        repre.feat_to_cluster_ids = cluster_ids
        repre.template_desc_opts = repre_util.TemplateDescOpts(desc_type="tfidf")
        print(f"Object {object_id}: building TF-IDF descriptors", flush=True)
        repre.template_descs, repre.feat_cluster_idfs = template_util.calc_tfidf_descriptors(
            feat_vectors=feat_vectors,
            feat_to_word_ids=repre.feat_to_cluster_ids,
            feat_to_template_ids=repre.feat_to_template_ids,
            feat_words=repre.feat_cluster_centroids,
            num_templates=len(repre.templates),
            tfidf_knn_k=repre.template_desc_opts.tfidf_knn_k,
            tfidf_soft_assign=repre.template_desc_opts.tfidf_soft_assign,
            tfidf_soft_sigma_squared=repre.template_desc_opts.tfidf_soft_sigma_squared,
        )
        repre.feat_vis_projectors = [pca]

        print(f"Object {object_id}: saving representation", flush=True)
        repre_dir.mkdir(parents=True, exist_ok=True)
        repre_util.save_object_repre(repre, str(repre_dir))

    def foundpose_inference(self, image, mask, camera, object_id):
        from aic_perception.utils.foundpose import corresp_util

        ys, xs = mask.nonzero()
        box = np.array(misc.calc_2d_box(xs, ys))
        box = structs.AlignedBox2f(
            left=box[0],
            top=box[1],
            right=box[2],
            bottom=box[3],
        )

        crop_box = misc.calc_crop_box(box=box, make_square=True)
        crop_camera = misc.construct_crop_camera(
            box=crop_box,
            camera_model_c2w=camera,
            viewport_size=self.crop_size,
            viewport_rel_pad=self.crop_rel_pad,
        )
        crop_image = misc.warp_image(
            src_camera=camera,
            dst_camera=crop_camera,
            src_image=image,
            interpolation=cv2.INTER_LINEAR,
        )
        crop_mask = misc.warp_image(
            src_camera=camera,
            dst_camera=crop_camera,
            src_image=mask,
            interpolation=cv2.INTER_NEAREST,
        )

        grid_points = feature_util.generate_grid_points(
            grid_size=self.crop_size,
            cell_size=self.grid_cell_size,
        ).to(self.device)

        with torch.inference_mode():
            image_tensor = (
                misc.array_to_tensor(crop_image)
                .to(torch.float32)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(self.device)
            )
            feature_map = self.extractor(image_tensor)["feature_maps"][0]
            query_points = feature_util.filter_points_by_mask(
                grid_points,
                misc.array_to_tensor(crop_mask).to(self.device),
            )
            query_features = feature_util.sample_feature_map_at_points(
                feature_map_chw=feature_map,
                points=query_points,
                image_size=self.crop_size,
            ).contiguous()

        object_data = self.objects[object_id]
        repre = object_data["repre"]
        query_features = projector_util.project_features(
            feat_vectors=query_features,
            projectors=repre.feat_raw_projectors,
        ).to(torch.float32).contiguous()

        corresps = corresp_util.establish_correspondences(
            query_points=query_points,
            query_features=query_features,
            object_repre=repre,
            template_matching_type="tfidf",
            template_knn_indices=object_data["template_indices"],
            feat_matching_type="cyclic_buddies",
            top_n_templates=self.match_top_n_templates,
            top_k_buddies=self.match_top_k_buddies,
            visual_words_knn_index=object_data["visual_words_index"],
        )

        poses = []
        for corresp_id, corresp in enumerate(corresps):
            if len(corresp["coord_2d"]) < 6:
                continue

            ok, R, t, inliers, quality = pnp_util.estimate_pose(
                corresp=corresp,
                camera_c2w=crop_camera,
                pnp_type="opencv",
                pnp_ransac_iter=self.pnp_ransac_iter,
                pnp_inlier_thresh=self.pnp_inlier_thresh,
                pnp_required_ransac_conf=0.99,
                pnp_refine_lm=True,
            )
            if not ok:
                continue

            T_m2crop = np.eye(4)
            T_m2crop[:3, :3] = R
            T_m2crop[:3, 3:] = t
            T_m2w = crop_camera.T_world_from_eye.dot(T_m2crop)
            T_m2c = np.linalg.inv(camera.T_world_from_eye).dot(T_m2w)
            poses.append(
                {
                    "object_id": object_id,
                    "R_m2c": T_m2c[:3, :3],
                    "t_m2c": T_m2c[:3, 3:],
                    "T_m2c": T_m2c,
                    "quality": quality,
                    "num_inliers": len(inliers),
                    "template_id": corresp["template_id"],
                    "corresp_id": corresp_id,
                }
            )

        return max(poses, key=lambda pose: pose["quality"]) if poses else None

    def alignpose_inference(self, camera_inputs, poses):
        candidates = []
        for camera_name, camera_results in poses.items():
            camera = self.camera_models[camera_name]
            for object_id, object_poses in camera_results.items():
                for pose in object_poses:
                    if pose is None:
                        continue
                    T_m2w = camera.T_world_from_eye.dot(pose["T_m2c"])
                    candidates.append(
                        {
                            "object_id": object_id,
                            "score": pose["quality"],
                            "T_m2w": T_m2w,
                        }
                    )

        if len(candidates) == 0:
            return poses

        candidate_rows = []
        for candidate in candidates:
            T_m2w = candidate["T_m2w"]
            candidate_rows.append(
                {
                    "obj_id": candidate["object_id"],
                    "score": candidate["score"],
                    "pose": structs.ObjectPose(R=T_m2w[:3, :3], t=T_m2w[:3, 3:]),
                    "candidate": candidate,
                }
            )
        object_dataset = SimpleNamespace(
            objects={
                object_id: SimpleNamespace(mesh=object_data["mesh"])
                for object_id, object_data in self.objects.items()
            }
        )
        kept_candidates = list(
            nms_3d_base.nms_candidates(
                candidates=pd.DataFrame(candidate_rows),
                object_dataset=object_dataset,
                iou_threshold=self.alignpose_nms_threshold,
                box_type=self.alignpose_nms_box_type,
                nms_mode=self.alignpose_nms_mode,
            )["candidate"]
        )

        refined_candidates = []
        for candidate in kept_candidates:
            object_id = candidate["object_id"]
            object_data = self.objects[object_id]
            repre = object_data["repre"]
            T_m2w = candidate["T_m2w"]
            cameras = []
            queries = []
            templates = []

            for camera_name, camera_input in camera_inputs.items():
                camera = self.camera_models[camera_name]
                image = camera_input["color"]
                T_m2c = np.linalg.inv(camera.T_world_from_eye).dot(T_m2w)
                vertices_c = object_data["vertices_np"].dot(T_m2c[:3, :3].T) + T_m2c[:3, 3]
                vertices_c = vertices_c[vertices_c[:, 2] > 0]
                if len(vertices_c) == 0:
                    continue

                uv = camera.eye_to_window(vertices_c)
                if uv[:, 0].max() < 0 or uv[:, 1].max() < 0:
                    continue
                if uv[:, 0].min() >= camera.width or uv[:, 1].min() >= camera.height:
                    continue

                box = structs.AlignedBox2f(
                    left=max(0, uv[:, 0].min()),
                    top=max(0, uv[:, 1].min()),
                    right=min(camera.width - 1, uv[:, 0].max()),
                    bottom=min(camera.height - 1, uv[:, 1].max()),
                )
                if box.width < 2 or box.height < 2:
                    continue

                crop_box = misc.calc_crop_box(box=box, make_square=True)
                crop_camera = misc.construct_crop_camera(
                    box=crop_box,
                    camera_model_c2w=camera,
                    viewport_size=self.crop_size,
                    viewport_rel_pad=self.crop_rel_pad,
                )
                crop_image = misc.warp_image(
                    src_camera=camera,
                    dst_camera=crop_camera,
                    src_image=image,
                    interpolation=cv2.INTER_LINEAR,
                )

                with torch.no_grad():
                    image_tensor = (
                        misc.array_to_tensor(crop_image)
                        .to(torch.float32)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    feature_map = self.extractor(image_tensor)["feature_maps"][0]
                    c, h, w = feature_map.shape
                    feature_map = projector_util.project_features(
                        feat_vectors=feature_map.permute(1, 2, 0).reshape(-1, c),
                        projectors=repre.feat_raw_projectors,
                    )
                    feature_map = feature_map.view(h, w, -1).permute(2, 0, 1).contiguous()

                T_m2crop = np.linalg.inv(crop_camera.T_world_from_eye).dot(T_m2w)
                rendered = self.renderer.render_object(
                    obj_id=object_id,
                    pose_m2c=T_m2crop,
                    camera_intrinsics=crop_camera,
                    render_types=[RenderType.COLOR, RenderType.DEPTH, RenderType.MASK],
                    return_tensors=True,
                )
                template_image = (
                    rendered[RenderType.COLOR]
                    .permute(2, 0, 1)
                    .contiguous()
                    .to(self.device)
                )
                template_depth = rendered[RenderType.DEPTH].to(self.device)
                template_mask = rendered[RenderType.MASK].to(self.device)
                T_model_from_crop = torch.inverse(
                    misc.array_to_tensor(T_m2crop).to(torch.float32).to(self.device)
                )

                with torch.no_grad():
                    template_features, _, template_vertices = feature_util.get_visual_features_registered_in_3d(
                        image_chw=template_image,
                        depth_image_hw=template_depth,
                        object_mask=template_mask,
                        camera=crop_camera,
                        T_model_from_camera=T_model_from_crop,
                        extractor=self.extractor,
                        grid_cell_size=self.grid_cell_size,
                    )
                if template_features.shape[0] == 0:
                    continue
                template_features = projector_util.project_features(
                    feat_vectors=template_features,
                    projectors=repre.feat_raw_projectors,
                ).to(torch.float32).contiguous()

                queries.append(
                    QueryRepre(
                        features=feature_map.reshape(feature_map.shape[0], -1).transpose(0, 1),
                        rgb=(crop_image * 255).astype(np.uint8),
                    )
                )
                templates.append(
                    TemplateRepre(
                        vertices=template_vertices,
                        masked_features=template_features,
                    )
                )
                cameras.append(crop_camera)

            if len(cameras) == 0:
                refined_candidates.append(candidate)
                continue

            initial_pose = structs.ObjectPose(
                R=T_m2w[:3, :3],
                t=T_m2w[:3, 3:],
            )
            refined_pose, failed, costs = refine_multiview_wrapper(
                initial_pose_wm=initial_pose,
                templates=templates,
                queries=queries,
                cameras=cameras,
                num_iter=self.alignpose_num_iters,
                loss_fn=self.alignpose_loss_fn,
            )
            if torch.as_tensor(failed).any().item():
                refined_candidates.append(candidate)
                continue

            T_refined_m2w = misc.get_rigid_matrix(refined_pose)
            score = candidate["score"]
            if costs is not None and len(costs) > 0:
                score = 1.0 - torch.mean(costs[-1]).detach().cpu().item()

            refined_candidates.append(
                {
                    "object_id": object_id,
                    "score": score,
                    "T_m2w": T_refined_m2w,
                }
            )

        results = {}
        for camera_name, camera_input in camera_inputs.items():
            camera = self.camera_models[camera_name]
            results[camera_name] = {}
            for object_id in camera_input["masks"]:
                results[camera_name][object_id] = []

            for candidate in refined_candidates:
                object_id = candidate["object_id"]
                if object_id not in results[camera_name]:
                    results[camera_name][object_id] = []
                T_m2c = np.linalg.inv(camera.T_world_from_eye).dot(candidate["T_m2w"])
                results[camera_name][object_id].append(
                    {
                        "object_id": object_id,
                        "R_m2c": T_m2c[:3, :3],
                        "t_m2c": T_m2c[:3, 3:],
                        "T_m2c": T_m2c,
                        "T_m2w": candidate["T_m2w"],
                        "quality": candidate["score"],
                        "num_inliers": None,
                        "template_id": None,
                        "corresp_id": None,
                    }
                )

        return results
