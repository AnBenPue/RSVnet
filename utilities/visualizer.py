import open3d as o3d

class visualizer:
    """
    The goal of this class is to simplify the visualizations through the 
    project and provide simple methods to add the common visualization aids
    such as frames or spheres.
    """
    def __init__(self):
        self.visualization = list()

    def addPointcloud(self,cloud:o3d.geometry.PointCloud,t=None,R=None,color=None):
        if color is not None:
            cloud.paint_uniform_color(color)
        if t is not None:
            cloud.translate(t)
        if R is not None:
            cloud.rotate(R)
        self.visualization.append(cloud)

    def addSphere(self, center, color=[0.5,0.5,0.5], radius=1.0):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
        sphere.paint_uniform_color(color)
        sphere.translate(center)
        self.visualization.append(sphere)

    def addLine(self, origin_point, end_point, color=[0,0,0]):        
        points = [origin_point, end_point]
        lines = [[0, 1]]
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        self.visualization.append(line_set)

    def addFrame(self, center, size=1, t=None, R=None):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size, center)
        if t is not None:
            frame.translate(t)
        if R is not None:
            frame.rotate(R)
        self.visualization.append(frame)

    def addArrow(self, cylinder_radius=1.0, cone_radius=1.5, cylinder_height=5.0, cone_height=4.0, resolution=20, cylinder_split=4, cone_split=1, t=None, R=None, color=[0,0,0]):
        arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius, cone_radius, cylinder_height, cone_height, resolution, cylinder_split, cone_split)
        arrow.paint_uniform_color(color)
        if t is not None:
            arrow.translate(t)
        if R is not None:
            arrow.rotate(R)
        self.visualization.append(arrow)

    def show(self, editing=False):
        if len(self.visualization) != 0:
            if editing is True:
                o3d.visualization.draw_geometries_with_editing(self.visualization)
            else:
                o3d.visualization.draw_geometries(self.visualization)
            self.visualization.clear()
        else:
            print('ERROR: Nothing to visualize')