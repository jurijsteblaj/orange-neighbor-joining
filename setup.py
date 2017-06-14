from setuptools import setup

setup(name="Orange3-NeighborJoining",
      packages=["neighborjoining"],
      package_data={"neighborjoining": ["icons/*.svg"]},
      classifiers=["Example :: Invalid"],
      entry_points={"orange.widgets": "Neighbor Joining = neighborjoining"},
      )
