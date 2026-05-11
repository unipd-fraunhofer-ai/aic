from setuptools import find_packages, setup

package_name = 'aic_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='iaslab',
    maintainer_email='matteo.terreran@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_node = aic_perception.simple_node:main',
            'test_camera_node = aic_perception.test_camera_node:main',
            'yolo_wrapper_node = aic_perception.yolo_wrapper_node:main',
            'pose_estimator_node = aic_perception.pose_estimator_node:main',
            'pose_estimator_mv_node = aic_perception.pose_estimator_mv_node:main',
        ],
    },
)
