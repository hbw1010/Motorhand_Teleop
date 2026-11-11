from setuptools import find_packages, setup

package_name = "motorhand_glove"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/udev", ["udev/90-motorhand-glove.rules"]),
    ],
    # The worker runs from this project's own Pixi environment; ROS2 only supplies the topic adapter.
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Motorhand FFB",
    maintainer_email="dev@example.com",
    description="Standalone Motorhand glove hand-following and force-feedback adapter",
    license="MIT",
    entry_points={"console_scripts": ["motorhand_glove = motorhand_glove.node:main"]},
)
