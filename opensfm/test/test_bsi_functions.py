import sys
from os.path import abspath, dirname, join

sys.path.insert(0, abspath(join(dirname("opensfm/"), "..")))
from opensfm import pybsi

# Simple dot product test case
a1 = [1, 2, 3]
a2 = [1,2,3]
print(pybsi.dot(a1,a2))


# Test mesh command
def triangle_mesh(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[Any], List[Any]]:
    """
    Create triangle meshes in a list
    """
    if shot_id not in r.shots or shot_id not in tracks_manager.get_shot_ids():
        return [], []

    shot = r.shots[shot_id]
    return triangle_mesh_perspective(shot_id, r, tracks_manager)


def triangle_mesh_perspective(
    shot_id: str, r: types.Reconstruction, tracks_manager: pymap.TracksManager
) -> Tuple[List[Any], List[Any]]:
    shot = r.shots[shot_id]
    cam = shot.camera

    dx = float(cam.width) / 2 / max(cam.width, cam.height)
    dy = float(cam.height) / 2 / max(cam.width, cam.height)
    pixels = [[-dx, -dy], [-dx, dy], [dx, dy], [dx, -dy]]
    vertices = [None for i in range(4)]
    for track_id in tracks_manager.get_shot_observations(shot_id):
        if track_id in r.points:
            point = r.points[track_id]
            pixel = shot.project(point.coordinates)
            nonans = not np.isnan(pixel).any()
            if nonans and -dx <= pixel[0] <= dx and -dy <= pixel[1] <= dy:
                vertices.append(point.coordinates)
                pixels.append(pixel.tolist())

    try:
        tri = scipy.spatial.Delaunay(pixels)
    except Exception as e:
        logger.error("Delaunay triangulation failed for input: {}".format(repr(pixels)))
        raise e

    sums = [0.0, 0.0, 0.0, 0.0]
    depths = [0.0, 0.0, 0.0, 0.0]
    for t in tri.simplices:
        for i in range(4):
            if i in t:
                for j in t:
                    if j >= 4:
                        depths[i] += shot.pose.transform(vertices[j])[2]
                        sums[i] += 1
    for i in range(4):
        if sums[i] > 0:
            d = depths[i] / sums[i]
        else:
            d = 50.0
        vertices[i] = back_project_no_distortion(shot, pixels[i], d).tolist()

    faces = tri.simplices.tolist()
    return vertices, faces

def back_project_no_distortion(
    shot: pymap.Shot, pixel: List[float], depth: float
) -> np.ndarray:
    """
    Back-project a pixel of a perspective camera ignoring its radial distortion
    """
    K = shot.camera.get_K()
    K1 = np.linalg.inv(K)
    p = np.dot(K1, [pixel[0], pixel[1], 1])
    p *= depth / p[2]
    return shot.pose.transform_inverse(p)