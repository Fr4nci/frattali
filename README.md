# Fractals: simple script for generating

## Introduction
The scripts in these repo were made to generate various types of fractals. For now you have to intervene manually to change the type of fractal obtained, but still everything is fully functional. The idea behind the generated fractals is as follows: we consider a sequence defined by recurrence of the following type

$$z_{n+1} = z_n^2 + c$$

with $\forall n \in \mathbb{N}, z_n  \in \mathbb{C}$. The behavior depends in particular on the constant $c$ considered: in the case of the set of Mandelbrot this is made to vary for every point of the complex plan, while in the case of the set of Julia the constant c is fixed. As for the BurningShip fractal, the absolute value of the real and imaginary part of the complex number is considered when the calculations are made, but from the point of view of "construction" the basic idea is always to fix the value of the constant c. 
To optimize the time I used the features of CUDA and Numpy to speed up the calculations.
Inside there are two scripts: the first serves to generate the set of Mandelbrot (in which $\forall z \in \mathbb{C}$ you vary the constant $c \in \mathbb{C}$), while with the other script you can go to generate the set of Julia to $ c$ constant, by modifying the code accordingly.
The logic remains equivalent even if you work in the set of $\mathbb{H}$ (the set of quaternions where each element is defined by an uplift of 4 real numbers). However, the need arises to "convert" the image from 4D to 3D in some way through two methods
1) We simply remove a dimension by "forgetting" one when we plot 
2) Otherwise by projecting from a practically isomorphic space to $\mathbb{R}^4$ to an isomorphic space to $\mathbb{R}^3$

For computational issues I preferred the first way, but at the moment I will not dwell on the mathematical details necessary to generate the image and, above all, implement in some way the distance between the fractal and the "camera" from which the scene is generated
## Requirement
To work this script requires the presence of the library _matplotlib_, _numpy_, _pillow_, _numba_ and requires all the tools that can be installed via the command
```bash
conda install cudatoolkit
```
## Some images generated by the scripts cointained in this repo
![mandelbrot_set](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/mandelbrot_set.png)
![julia_set](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/julia_set.png)
![julia_set_2](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/julia_colored.png)
![burning_ship](https://github.com/Fr4nci/frattali/blob/main/Immagini%20varie%20generate/immagine_zoom_burning_ship.png)
![4d_julia_2](https://github.com/Fr4nci/frattali/blob/main/Frattali%20in%204D/frattale4_3.png)
![4d_julia_3](https://github.com/Fr4nci/frattali/blob/main/Frattali%20in%204D/frattale5_1.png) 

# Note
For those interested in the zoom implementation, I basically used the available matplotlib library. You will see that the coordinates _y_max_ and _y_min_ in the generation of the new fractal are reversed.
I also tried to write an equivalent program using the CUDA language (C++ modified in a proprietary way by Nvidia to allow parallel computing)
