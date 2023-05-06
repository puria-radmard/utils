from purias_utils.geometry.topology import Manifold, Chart, UnitRightRectangle, RightRectangle
from purias_utils.geometry.standard.charts import *
from torch import tensor
from math import pi



class StandardS1(Manifold):

    def __init__(self, grid_count):

        chart1 = Chart(
            f=generate_offset_chart_s1(0.0),
            f_inverse=generate_offset_chart_s1(-0.0),
            domain=RightRectangle(left=tensor([0.0]), right=tensor([pi])),
            codomain=RightRectangle(left=tensor([0.0]), right=tensor([pi])),
        )

        chart2 = Chart(
            f=generate_offset_chart_s1(pi),
            f_inverse=generate_offset_chart_s1(-pi),
            domain=RightRectangle(left=tensor([pi]), right=tensor([2 * pi])),
            codomain=RightRectangle(left=tensor([0.0]), right=tensor([pi])),
        )

        charts=[chart1, chart2]
        set_left=tensor([0.0])
        set_right=tensor([2 * pi])

        super().__init__(set_left, set_right, charts, grid_count)



class StandardCylinder(Manifold):

    def __init__(self, grid_count):

        chart1 = Chart(
            f=generate_offset_cylinder_chart(0.0),
            f_inverse=generate_offset_cylinder_inverse_chart(0.0),
            domain=RightRectangle(left=tensor([0.0, 0.0]), right=tensor([pi, 1.0])),
            codomain=UnitRightRectangle(2),
        )

        chart2 = Chart(
            f=generate_offset_cylinder_chart(pi),
            f_inverse=generate_offset_cylinder_inverse_chart(pi),
            domain=RightRectangle(left=tensor([pi, 0.0]), right=tensor([2 * pi, 1.0])),
            codomain=UnitRightRectangle(2),
        )

        charts = [chart1, chart2]
        set_left=tensor([0.0, 0.0])
        set_right=tensor([2 * pi, 1.0])

        super().__init__(set_left, set_right, charts, grid_count)



class StandardS2(Manifold):

    def __init__(self, grid_count):

        chart1 = Chart(
            f=generate_offset_s2_chart(0.0),
            f_inverse=generate_offset_s2_inverse_chart(0.0),
            domain=RightRectangle(left=tensor([0.0, 0.0]), right=tensor([pi, pi])),
            codomain=UnitRightRectangle(2),
        )

        chart2 = Chart(
            f=generate_offset_s2_chart(pi),
            f_inverse=generate_offset_s2_inverse_chart(pi),
            domain=RightRectangle(left=tensor([0.0, pi]), right=tensor([pi, 2 * pi])),
            codomain=UnitRightRectangle(2),
        )

        charts = [chart1, chart2]
        set_left=tensor([0.0, 0.0])
        set_right=tensor([pi, 2 * pi])

        super().__init__(set_left, set_right, charts, grid_count)
