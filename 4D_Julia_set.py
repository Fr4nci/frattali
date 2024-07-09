import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, njit, int16
from math import cos, sin, sqrt, log, pi

@cuda.jit(device=True)
def distance_estimate(z, c, max_iterations):
    magnitude_squared = z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2
    dz_squared = 1.0
    iteration_count = int16(max_iterations)
    for i in range(max_iterations):
        dz_squared *= 4 * magnitude_squared
        new_z = cuda.local.array(4, dtype=np.float32)
        new_z[0] = z[0]**2 - z[1]**2 - z[2]**2 - z[3]**2 + c[0]
        new_z[1] = 2 * z[0] * z[1] + c[1]
        new_z[2] = 2 * z[0] * z[2] + c[2]
        new_z[3] = 2 * z[0] * z[3] + c[3]
        z[0] = new_z[0]
        z[1] = new_z[1]
        z[2] = new_z[2]
        z[3] = new_z[3]
        magnitude_squared = z[0]**2 + z[1]**2 + z[2]**2 + z[3]**2
        if magnitude_squared > 1e7:
            iteration_count = int16(i)
            break
    return 0.5 * sqrt(magnitude_squared / dz_squared) * log(magnitude_squared), iteration_count

@cuda.jit(device=True)
def ray_march(azimuth_angle, altitude_angle, initial_position, max_steps, c_value, max_iterations, height, cutting_direction, out_current_position):
    direction_vector = (cos(azimuth_angle) * cos(altitude_angle), sin(azimuth_angle), -sin(altitude_angle))
    norm_direction = sqrt(direction_vector[0]**2 + direction_vector[1]**2 + direction_vector[2]**2)
    distance, _ = distance_estimate(initial_position, c_value, max_iterations)
    distance *= 0.5
    current_position = cuda.local.array(3, dtype=np.float32)
    current_position[0] = initial_position[0]
    current_position[1] = initial_position[1]
    current_position[2] = initial_position[2]
    temp_height = height * sqrt(1 - sin(azimuth_angle)**2 * sin(altitude_angle)**2) / sin(azimuth_angle)
    current_position[0] -= direction_vector[0] * temp_height / direction_vector[cutting_direction]
    current_position[1] -= direction_vector[1] * temp_height / direction_vector[cutting_direction]
    current_position[2] -= direction_vector[2] * temp_height / direction_vector[cutting_direction]
    
    for step in range(max_steps):
        if distance < 1e-7:  # Closest approach distance
            out_current_position[0] = current_position[0]
            out_current_position[1] = current_position[1]
            out_current_position[2] = current_position[2]
            return True, sqrt((current_position[0] - initial_position[0])**2 + (current_position[1] - initial_position[1])**2 + (current_position[2] - initial_position[2])**2), step, distance_estimate(current_position, c_value, max_iterations)[1]
        elif sqrt((current_position[0] - initial_position[0])**2 + (current_position[1] - initial_position[1])**2 + (current_position[2] - initial_position[2])**2) > 50 or distance > 50:
            out_current_position[0] = 0.0
            out_current_position[1] = 0.0
            out_current_position[2] = 0.0
            return False, sqrt((current_position[0] - initial_position[0])**2 + (current_position[1] - initial_position[1])**2 + (current_position[2] - initial_position[2])**2), step, distance_estimate(current_position, c_value, max_iterations)[1]
        else:
            current_position[0] += distance / norm_direction * direction_vector[0]
            current_position[1] += distance / norm_direction * direction_vector[1]
            current_position[2] += distance / norm_direction * direction_vector[2]
            distance, _ = distance_estimate(current_position, c_value, max_iterations)
            distance *= 0.5
    out_current_position[0] = current_position[0]
    out_current_position[1] = current_position[1]
    out_current_position[2] = current_position[2]
    return False, sqrt((current_position[0] - initial_position[0])**2 + (current_position[1] - initial_position[1])**2 + (current_position[2] - initial_position[2])**2), max_steps, distance_estimate(current_position, c_value, max_iterations)[1]

@cuda.jit
def fractal_kernel(c_value, initial_position, height, cutting_direction, azimuth_grid, altitude_grid, resolution_x, resolution_y, max_ray_steps, max_iterations, hit_matrix, distance_matrix, iteration_matrix, normals_matrix, escape_time_matrix):
    i, j = cuda.grid(2)
    if i < resolution_y and j < resolution_x:
        out_current_position = cuda.local.array(3, dtype=np.float32)
        result = ray_march(azimuth_grid[i], altitude_grid[j], initial_position, max_ray_steps, c_value, max_iterations, height, cutting_direction, out_current_position)
        hit_matrix[i, j] = result[0]
        distance_matrix[i, j] = result[1] if result[1] < 20 else 20.0
        iteration_matrix[i, j] = result[2]
        normals_matrix[i, j, 0] = out_current_position[0]
        normals_matrix[i, j, 1] = out_current_position[1]
        normals_matrix[i, j, 2] = out_current_position[2]
        escape_time_matrix[i, j] = result[3]

def fractal(c_value, initial_position, height, cutting_direction):
    resolution_x = 4000
    resolution_y = 4000
    azimuth_span = 35
    max_ray_steps = 4000
    max_iterations = 200

    azimuth_grid = np.linspace((-azimuth_span * resolution_y / resolution_x + 93) * pi / 180, (azimuth_span * resolution_y / resolution_x + 93) * pi / 180, resolution_y)
    altitude_grid = np.linspace((-azimuth_span + 12) * pi / 180, (azimuth_span + 12) * pi / 180, resolution_x)

    hit_matrix = np.zeros((resolution_y, resolution_x), dtype=np.bool_)
    distance_matrix = np.zeros((resolution_y, resolution_x), dtype=np.float32)
    iteration_matrix = np.zeros((resolution_y, resolution_x), dtype=np.int16)
    normals_matrix = np.zeros((resolution_y, resolution_x, 3), dtype=np.float32)
    escape_time_matrix = np.zeros((resolution_y, resolution_x), dtype=np.int16)

    d_c_value = cuda.to_device(c_value)
    d_initial_position = cuda.to_device(initial_position)
    d_azimuth_grid = cuda.to_device(azimuth_grid)
    d_altitude_grid = cuda.to_device(altitude_grid)
    d_hit_matrix = cuda.to_device(hit_matrix)
    d_distance_matrix = cuda.to_device(distance_matrix)
    d_iteration_matrix = cuda.to_device(iteration_matrix)
    d_normals_matrix = cuda.to_device(normals_matrix)
    d_escape_time_matrix = cuda.to_device(escape_time_matrix)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(resolution_y / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(resolution_x / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    fractal_kernel[blockspergrid, threadsperblock](d_c_value, d_initial_position, height, cutting_direction, d_azimuth_grid, d_altitude_grid, resolution_x, resolution_y, max_ray_steps, max_iterations, d_hit_matrix, d_distance_matrix, d_iteration_matrix, d_normals_matrix, d_escape_time_matrix)

    hit_matrix = d_hit_matrix.copy_to_host()
    distance_matrix = d_distance_matrix.copy_to_host()
    iteration_matrix = d_iteration_matrix.copy_to_host()
    normals_matrix = d_normals_matrix.copy_to_host()
    escape_time_matrix = d_escape_time_matrix.copy_to_host()

    distance_matrix = 1.0 - distance_matrix / 20
    return hit_matrix, iteration_matrix, distance_matrix, normals_matrix, escape_time_matrix

def color_map(value, color_scheme):
    if color_scheme == "grayscale":
        return np.array([value, value, value])
    elif color_scheme == "hot":
        return np.array([value, value**3, value**6])
    elif color_scheme == "cold":
        return np.array([value**6, value**3, value])
    else:
        return np.array([value, value, value])

def apply_color_map(distance_matrix, color_scheme):
    color_matrix = np.zeros((distance_matrix.shape[0], distance_matrix.shape[1], 3))
    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            color_matrix[i, j] = color_map(distance_matrix[i, j], color_scheme)
    return color_matrix

def main():
    c_value = np.array([-0.75, 0.1, 0.0, 0.0], dtype=np.float32)
    initial_position = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    height = 5.0
    cutting_direction = 2

    hit_matrix, iteration_matrix, distance_matrix, normals_matrix, escape_time_matrix = fractal(c_value, initial_position, height, cutting_direction)

    color_scheme = "hot"
    colored_distance_matrix = apply_color_map(distance_matrix, color_scheme)

    plt.imshow(colored_distance_matrix)
    plt.show()

if __name__ == "__main__":
    main()

# Notebook di valori cool per c-Values
# c_value = Quaternion(-0.517518, -0.341729, -0.407854, -0.0716855)
# c_value = Quaternion(0.415189, 0.560351, 0.174757, 0.459138)
# c_value = Quaternion(0.365658, -0.0599171, -0.396929, -0.544048)
# c_value = Quaternion(-0.254991, -0.710382, -0.110794, 0.264363)  # Set di Julia principale
# c_value = Quaternion(-0.250349, -0.322733, -0.667216, -0.181973)
# c_value = Quaternion(0.20848, 0.0289175, -0.989516, 0.49807)
# c_value = Quaternion(-0.214601, -0.849091, 0.0252976, -0.065572)
# c_value = Quaternion(-0.577406, 0.167387, -0.482833, 0.0)
# c_value = Quaternion(-0.142171, 0.231055, -0.91313, -0.0292274)
# c_value = Quaternion(-0.234613, -0.276556, -0.876272, -0.0177794)
# c_value = Quaternion(-0.423564, -0.691938, -0.294925, 0.0715491)