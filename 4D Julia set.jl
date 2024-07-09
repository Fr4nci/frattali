using Images
using ReferenceFrameRotations
using ProgressBars
using LinearAlgebra
using Profile
using PProf 

function distance_estimate(z, c, max_iterations)
    magnitude_squared = norm(z)^2
    dz_squared = 1.0
    iteration_count = Int16(max_iterations)
    for i in 1:max_iterations
        dz_squared *= 4 * magnitude_squared
        z = z * z + c
        magnitude_squared = norm(z)^2
        if magnitude_squared > 1e7
            iteration_count = Int16(i)
            break
        end
    end
    return 0.5 * sqrt(magnitude_squared / dz_squared) * log(magnitude_squared), iteration_count
end

function ray_march(azimuth_angle, altitude_angle, initial_position, max_steps, c_value, max_iterations, height, cutting_direction)
    direction_vector = (cos(azimuth_angle) * cos(altitude_angle), sin(azimuth_angle), -sin(altitude_angle))
    norm_direction = norm(direction_vector)
    distance = 0.5 * distance_estimate(initial_position, c_value, max_iterations)[1]
    current_position = initial_position
    temp_height = height * sqrt(1 - sin(azimuth_angle)^2 * sin(altitude_angle)^2) / sin(azimuth_angle)
    current_position = Quaternion(
        current_position.q0 - direction_vector[1] * temp_height / direction_vector[cutting_direction],
        current_position.q1 - direction_vector[2] * temp_height / direction_vector[cutting_direction],
        current_position.q2 - direction_vector[3] * temp_height / direction_vector[cutting_direction],
        current_position.q3
    ) 
    for step in 1:max_steps
        if distance < 1e-7  # Distanza di approccio più vicina
            return true, sqrt((current_position.q0 - initial_position.q0)^2 + (current_position.q1 - initial_position.q1)^2 + (current_position.q2 - initial_position.q2)^2), step, [Float32(current_position.q0), Float32(current_position.q1), Float32(current_position.q2)]::Vector{Float32}, distance_estimate(current_position, c_value, max_iterations)[2]
        elseif sqrt((current_position.q0 - initial_position.q0)^2 + (current_position.q1 - initial_position.q1)^2 + (current_position.q2 - initial_position.q2)^2) > 50 || distance > 50
            return false, sqrt((current_position.q0 - initial_position.q0)^2 + (current_position.q1 - initial_position.q1)^2 + (current_position.q2 - initial_position.q2)^2), step, zeros(Float32, 3), distance_estimate(current_position, c_value, max_iterations)[2]
        else  
            current_position = Quaternion(
                current_position.q0 + distance / norm_direction * direction_vector[1],
                current_position.q1 + distance / norm_direction * direction_vector[2],
                current_position.q2 + distance / norm_direction * direction_vector[3],
                current_position.q3
            )
            distance = 0.5 * distance_estimate(current_position, c_value, max_iterations)[1] 
        end
    end
    return false, sqrt((current_position.q0 - initial_position.q0)^2 + (current_position.q1 - initial_position.q1)^2 + (current_position.q2 - initial_position.q2)^2), max_steps, [Float32(current_position.q0), Float32(current_position.q1), Float32(current_position.q2)]::Vector{Float32}, distance_estimate(current_position, c_value, max_iterations)[2]
end

function fractal(c_value, initial_position, height, cutting_direction)
    hit_matrix = falses(resolution_y, resolution_x)
    distance_matrix = zeros(resolution_y, resolution_x)
    iteration_matrix = zeros(resolution_y, resolution_x)
    normals_matrix = fill(Vector{Float32}(undef, 3), resolution_y, resolution_x)
    escape_time_matrix = zeros(Int16, resolution_y, resolution_x)
    Threads.@threads for i in ProgressBar(1:resolution_y)
        for j in 1:resolution_x
            result = ray_march(azimuth_grid[i], altitude_grid[j], initial_position, max_ray_steps, c_value, max_iterations, height, cutting_direction)
            hit_matrix[i, j] = result[1]
            result[2] < 20 ? distance_matrix[i, j] = result[2] : distance_matrix[i, j] = 20.0
            iteration_matrix[i, j] = result[3]
            normals_matrix[i, j] = result[4]
            escape_time_matrix[i, j] = result[5]
        end
    end
    distance_matrix = 1 .- distance_matrix / 20
    return hit_matrix, iteration_matrix, distance_matrix, normals_matrix, escape_time_matrix
end

######################## vv CONTROL PARAMETERS vv ########################

# Risoluzione dell'immagine
resolution_x = 4000
resolution_y = 4000

# Parametri di controllo
azimuth_span = 35               # Span angolare come lunghezza focale inversa
max_ray_steps = 4000            # Numero massimo di passi del raggio
max_iterations = 200            # Numero massimo di iterazioni nella stima della distanza
intersection_height = 0.0       # Altezza dell'intersezione
cutting_axis = 2                # Asse di taglio

# Angoli della telecamera
azimuth_grid = collect(range((-azimuth_span * resolution_y / resolution_x + 93) * π / 180, stop=(azimuth_span * resolution_y / resolution_x + 93) * π / 180, length=resolution_y))
altitude_grid = collect(range((-azimuth_span + 12) * π / 180, stop=(azimuth_span + 12) * π / 180, length=resolution_x))

# Posizione della telecamera
camera_position = Quaternion(0.0, -2.0, 0.25, 0.0)
camera_position_vector = [Float32(camera_position.q0), Float32(camera_position.q1), Float32(camera_position.q2)]

# Valore di c per il set di Julia
c_value = Quaternion(0.20848, 0.0289175, -0.989516, 0.49807)

# Percorso per salvare le immagini
image_path = "D:/Dati Windows/Documents/Frattali/"

######################## ^^ CONTROL PARAMETERS ^^ ########################

# Esegui il ray marching per il set di Julia
results = fractal(c_value, camera_position, intersection_height, cutting_axis)

# Calcola le normali
normals_matrix = fill(zeros(Float32, 3), resolution_y, resolution_x)
for i in ProgressBar(2:resolution_y-1)
    Threads.@threads for j in 2:resolution_x-1
        a = zeros(Float32, 3)
        b = zeros(Float32, 3)
        if abs(norm(results[4][i, j] - results[4][i, j-1])) < abs(norm(results[4][i, j] - results[4][i, j+1]))
            a = results[4][i, j-1]
        else
            a = results[4][i, j+1]
        end
        if abs(norm(results[4][i, j] - results[4][i-1, j])) < abs(norm(results[4][i, j] - results[4][i+1, j]))
            b = results[4][i-1, j]
        else
            b = results[4][i+1, j]
        end
        if norm(a) > 0
            normals_matrix[i, j] = normalize(cross(a - results[4][i, j], b - results[4][i, j]))
        end
    end
end

# Calcola l'illuminazione
shadow_matrix = zeros(Float32, resolution_y, resolution_x)
light_point = [Float32(-1), Float32(-2), Float32(2)]
Threads.@threads for i in ProgressBar(2:resolution_y-1)
    for j in 2:resolution_x-1
        if results[1][i, j] == 1.0
            light_direction = normalize(light_point - results[4][i, j])
            view_direction = normalize(camera_position_vector - results[4][i, j])
            halfway_direction = normalize(light_direction + view_direction)
            shadow_matrix[i, j] = 0.15 + 0.25 * (dot(light_direction, normals_matrix[i, j])) + (dot(halfway_direction, normals_matrix[i, j]))^10
        end
    end
end

shadow_matrix = max.(shadow_matrix / findmax(shadow_matrix)[1], 0.0)
save("$image_path/a.png", map(clamp01nan, RGB.((results[1] .* (2 ./ results[2])).^(1/4) .* (5 .+ shadow_matrix) / 6 , 0.15 * sqrt.(results[3] .+ results[1] ./ results[2]) .* (5 .+ shadow_matrix) / 6, 0.25 * sqrt.(results[3] .+ results[1] ./ results[2]) .* (5 .+ shadow_matrix) / 6)))
save("$image_path/b.png", map(clamp01nan, RGB.(sqrt.(results[1] .* (2 ./ results[2])) * 2 , 0.15 * sqrt.(results[3]) .+ sqrt.(results[3]) .* (2 ./ results[2]), 0.35 * sqrt.(results[3]) .+ sqrt.(results[3]) .* (2 ./ results[2]))))
save("$image_path/c.png", map(clamp01nan, RGB.(0.2 * sqrt.(results[1] .* (2 ./ results[2])) * 2, 0.8 * sqrt.(results[1] .* (2 ./ results[2])) * 2, 0.3 * sqrt.(results[1] .* (2 ./ results[2])) * 2 .+ results[3] / 2)))
save("$image_path/d.png", map(clamp01nan, RGB.(sqrt.(2 ./ results[2]) * 2 .* (results[3]).^13 ./ findmax((results[3]).^13)[1], 0.1 * (results[3]).^(4) .+ 0.45 * results[1] .* sqrt.(2 ./ results[2]), 0.35 * (results[3]).^(4) .+ 0.35 * results[1] .* sqrt.(2 ./ results[2]))))
save("$image_path/e.png", map(clamp01nan, 2 .* RGB.(110 * sqrt.(results[1] ./ results[2] .* (1 .- results[3]).^4) .* sqrt.(shadow_matrix), 0.6 .* sqrt.(results[1] .* (2 ./ results[2])) .* sqrt.(shadow_matrix), 0.7 .* (results[1] .* (2 ./ results[2])).^(1/3) .* sqrt.(shadow_matrix))))
save("$image_path/f.png", map(clamp01nan, RGB.(sqrt.(results[2] ./ findmax(results[2])[1]).^(1/2) .* ((results[3] ./ findmax(results[3])[1]).^12), 0.1 .* (results[3] / findmax(results[3])[1]).^6, (results[2] ./ findmax(results[2])[1]).^(1/2) .* (1 .- (results[3] ./ findmax(results[3])[1]).^12))))
save("$image_path/g.png", map(clamp01nan, RGB.(results[2] ./ maximum(results[2]), (results[1] .* results[3].^5 .- 1.5 * (results[5] ./ maximum(results[5])).^2) .* shadow_matrix .+ results[2] ./ maximum(results[2]), ((results[5] ./ maximum(results[5])).^0.8 .+ (1 .- (results[1] .* results[3].^5))) .* shadow_matrix / 2 .+ results[2] ./ maximum(results[2]))))

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

#### BONUS OPTIONS

canvas = Array{RGB{Float64}}(undef, resolution_y, resolution_x)
reference_colors = [
    RGB(158/255, 1/255, 66/255), RGB(215/255, 65/255, 78/255), RGB(237/255, 98/255, 70/255), RGB(249/255, 142/255, 82/255),
    RGB(253/255, 187/255, 108/255), RGB(254/255, 225/255, 141/255), RGB(1, 250/255, 182/255), RGB(239/255, 249/255, 166/255),
    RGB(205/255, 235/255, 157/255), RGB(177/255, 223/255, 163/255), RGB(148/255, 212/255, 164/255), RGB(126/255, 204/255, 165/255),
    RGB(102/255, 194/255, 165/255), RGB(84/255, 174/255, 173/255), RGB(68/255, 113/255, 178/255), RGB(91/255, 83/255, 164/255)
]
for (index, element) in ProgressBar(enumerate(results[5]))
    if element == 0.0
        canvas[index] = RGB(0, 0, 0)
    else
        canvas[index] = reference_colors[Int(round(0.35 * element + 14) % 16) + 1]
    end
end
# save("$image_path/color_pass.png", canvas .* results[1])
# save("$image_path/h.png", map(clamp01nan, canvas .* (results[1] .* sqrt.(shadow_matrix) .+ results[2] / findmax(results[2])[1] .+ 0.15 * results[3].^5)))

normals_r = zeros(resolution_y, resolution_x)
normals_g = zeros(resolution_y, resolution_x)
normals_b = zeros(resolution_y, resolution_x)

Threads.@threads for i in ProgressBar(1:resolution_x * resolution_y)
    normals_r[i] = results[4][i][1]
    normals_g[i] = results[4][i][2]
    normals_b[i] = results[4][i][3]
end

# save("$image_path/normals_pass.png", RGB.(map(clamp01nan, normals_r), map(clamp01nan, normals_g), map(clamp01nan, normals_b)))
