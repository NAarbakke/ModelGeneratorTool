"""Ground-truth parts for synthetic scans and the model benchmark.

Each part sits on the board (min z = 0), centered on the origin, in mm. Together
they cover every analytic surface type the fitter handles: planes, cylinders,
cones (chamfers), tori (fillets on round edges) and spheres.
"""

PARTS = {
    "block_hole": """
result = (cq.Workplane("XY").box(80, 50, 30, centered=(True, True, False))
          .edges("|Z").fillet(6)
          .faces(">Z").workplane().hole(20))
""",
    "flange": """
base = cq.Workplane("XY").circle(45).extrude(12)
boss = cq.Workplane("XY").workplane(offset=12).circle(20).extrude(18)
result = (base.union(boss)
          .faces(">Z").workplane().hole(16)
          .faces("<Z").workplane().polarArray(32.5, 45, 360, 4).hole(8))
""",
    "bracket": """
profile = (cq.Workplane("XZ").polyline([(0, 0), (60, 0), (60, 10), (10, 10), (10, 45), (0, 45)])
           .close())
result = (profile.extrude(-40).edges("|Y and >X and <Z").fillet(4)
          .faces(">Z").workplane().center(0, 0).hole(8)
          .translate((-30, -20, 0)))
""",
    "knob": """
result = (cq.Workplane("XY").circle(22).extrude(28)
          .faces(">Z").edges().fillet(7)
          .faces("<Z").edges().chamfer(3)
          .union(cq.Workplane("XY").workplane(offset=28).sphere(9)))
""",
}
