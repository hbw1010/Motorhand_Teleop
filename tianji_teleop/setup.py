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
    # The glove chain's own dependencies live in the Motorhand_Teleop checkout's Pixi env (worker.py).
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="tianji_teleop",
    maintainer_email="dev@example.com",
    description="Motorhand exoskeleton gloves to Wuji Hand 2 joint commands",
    license="MIT",
    entry_points={"console_scripts": ["motorhand_glove = motorhand_glove.node:main"]},
)
