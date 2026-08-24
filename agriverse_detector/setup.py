from setuptools import find_packages, setup

package_name = "agriverse_detector"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            "share/" + package_name,
            ["package.xml"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    description="ROS 2 crop and weed detector using YOLO.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "detector_node = agriverse_detector.detector_node:main",
        ],
    },
)
