using Images
using ReferenceFrameRotations
using ProgressBars
using LinearAlgebra
using Profile
using PProf 

function dist(z,c, iter) #Distance Estimation
    m2=norm(z)^2
    dz2=1.0
    tmp=Int16(iter)
    for i in 1:iter
        dz2*=4*m2
        z=z*z+c
        m2=norm(z)^2
        if m2>1e7
            tmp=Int16(i)
            break
        end
    end
    return 0.5*sqrt(m2/dz2)*log(m2),tmp
end

function steps(azi, alt, P0,steps,c,iter,hgt,cut_dir) #Raymarch ray of given initial angels, return [hit], [dist from origin], [#steps] & [final ray position]
    vek=(cos(azi)*cos(alt), sin(azi), -sin(alt))
    nrm=norm(vek)
    d=0.5*dist(P0,c,iter)[1]
    P=P0
    tmp=hgt*sqrt(1-sin(azi)^2*sin(alt)^2)/sin(azi)
    P=Quaternion(P.q0-vek[1]*tmp/vek[cut_dir],P.q1-vek[2]*tmp/vek[cut_dir],P.q2-vek[3]*tmp/vek[cut_dir],P.q3) 
    for i in 1:steps
        if d < 10^(-7) #closest approach distance
            return true, sqrt((P.q0-P0.q0)^2+(P.q1-P0.q1)^2+(P.q2-P0.q2)^2),i,[Float32(P.q0),Float32(P.q1),Float32(P.q2)]::Vector{Float32},dist(P,c,iter)[2]
        elseif sqrt((P.q0-P0.q0)^2+(P.q1-P0.q1)^2+(P.q2-P0.q2)^2) > 50 || d>50
            return false,sqrt((P.q0-P0.q0)^2+(P.q1-P0.q1)^2+(P.q2-P0.q2)^2),i,zeros(Float32,3),dist(P,c,iter)[2]
        else  
            P=Quaternion(P.q0+d/nrm*vek[1],P.q1+d/nrm*vek[2],P.q2+d/nrm*vek[3],P.q3)
            d=0.5*dist(P,c,iter)[1] #0.5* to eliminate overshooting from DE 
        end
    end
    return false,sqrt((P.q0-P0.q0)^2+(P.q1-P0.q1)^2+(P.q2-P0.q2)^2),steps,[Float32(P.q0),Float32(P.q1),Float32(P.q2)]::Vector{Float32},dist(P,c,iter)[2]
end

function frac(c,P,hgt,cut_dir) #MAIN
    hit_mt=falses(resy,resx)
    dist_mt=zeros(resy,resx)
    iter_mt=zeros(resy,resx)
    nrm_mt=fill(Vector{Float32}(undef,3),resy,resx)
    et_mt=zeros(Int16,resy,resx)
    Threads.@threads for i in ProgressBar(1:resy)
        for j in 1:resx
            a=steps(grid_azi[i], grid_alt[j], P, max_steps,c,max_iter,hgt,cut_dir)
            hit_mt[i,j]=a[1]
            a[2] < 20 ? dist_mt[i,j]=a[2] : dist_mt[i,j]=20. 
            iter_mt[i,j]=a[3]
            nrm_mt[i,j]=a[4]
            et_mt[i,j]=a[5]
        end
    end
    dist_mt=1 .-dist_mt/20
    return hit_mt, iter_mt, dist_mt, nrm_mt,et_mt
end


######################## vv CONTROL PARAMETERS vv ########################


#Resolution
resx=4000
resy=4000
#resx=1000
#resy=1000

azi=35              # Like an inverse focal length
max_steps=4000      # Maximum number of Ray Steps
max_iter=200        # Maximum number of iterations in Distance Estimation
hgt=0.              # Height of intersection
cut_dir=2           # Change to cut along different axis

# Camera angel
grid_azi=collect(range((-azi*resy/resx+93)*π/180, stop=(azi*resy/resx+93)*π/180, length=resy))  #Up-Down: lower number -> pan up
grid_alt=collect(range((-azi+12)*π/180, stop=(azi+12)*π/180, length=resx))                            #Left-Right

# Camera Position
P=Quaternion(0.,-2.,0.25,0.)
P3=[Float32(P.q0),Float32(P.q1),Float32(P.q2)]::Vector{Float32}
# c-Value of Julia set
c=Quaternion(- 0.254991, - 0.710382, - 0.110794, + 0.264363)
#c=Quaternion(rand()*2-1,rand()*2-1,rand()*2-1,rand()*0.2-0.1) #Randomly generate c-value

imgpath="D:/Dati Windows/Documents/Frattali/"
######################## ^^ CONTROL PARAMETERS ^^ ########################

jl=frac(c,P,hgt,cut_dir) #Raymarch Julia set
#map(clamp01nan,RGB.(110*sqrt.(jl[1] ./jl[2] .* (1 .- jl[3]).^4.),0.6.*sqrt.(jl[1] .* (2 ./jl[2])),0.7.*(jl[1] .* (2 ./jl[2])).^(1/3)))
#save("$imgpath/e_r.png", map(clamp01nan,2 .*RGB.(110*sqrt.(jl[1] ./jl[2] .* (1 .- jl[3]).^4.),0.6.*sqrt.(jl[1] .* (2 ./jl[2])),0.7.*(jl[1] .* (2 ./jl[2])).^(1/3))))
nrm_mt=fill(zeros(Float32, 3),resy,resx)

for i in ProgressBar(2:resy-1)
    Threads.@threads for j in 2:resx-1
        a=zeros(Float32,3);b=zeros(Float32,3)
        abs(norm(jl[4][i,j]-jl[4][i,j-1]))<abs(norm(jl[4][i,j]-jl[4][i,j+1])) ? a=jl[4][i,j-1] : a=jl[4][i,j+1]
        abs(norm(jl[4][i,j]-jl[4][i-1,j]))<abs(norm(jl[4][i,j]-jl[4][i+1,j])) ? b=jl[4][i-1,j] : b=jl[4][i+1,j]
        norm(a)>0. ? nrm_mt[i,j]=normalize(cross(a-jl[4][i,j],b-jl[4][i,j])) : nothing
    end
end

shdw_mt=zeros(Float32, resy,resx)
light_pt=[Float32(-1),Float32(-2),Float32(2)]
Threads.@threads for i in ProgressBar(2:resy-1)
    for j in 2:resx-1
        if jl[1][i,j]==1.
            light_dir=normalize(light_pt-jl[4][i,j])
            view_dir=normalize(P3-jl[4][i,j])
            halfway_dir=normalize(light_dir+view_dir)
            shdw_mt[i,j]=0.15+0.25*(dot(light_dir,nrm_mt[i,j]))+(dot(halfway_dir,nrm_mt[i,j]))^10
        end
    end
end

shdw_mt=max.(shdw_mt/findmax(shdw_mt)[1],0.)
#map(clamp01nan,Gray.(shdw_mt))
#save("$imgpath/lighting_pass.png",map(clamp01nan,Gray.(max.(shdw_mt,0.)/findmax(shdw_mt)[1])))
save("$imgpath/a.png",map(clamp01nan,RGB.((jl[1] .* (2 ./jl[2])).^(1/4).*(5 .+shdw_mt)/6 , 0.15*sqrt.(jl[3].+jl[1]./jl[2]).*(5 .+shdw_mt)/6, 0.25*sqrt.(jl[3].+jl[1]./jl[2]).*(5 .+shdw_mt)/6)))
save("$imgpath/b.png", map(clamp01nan,RGB.(sqrt.(jl[1] .* (2 ./jl[2]))*2 , 0.15*sqrt.(jl[3]).+sqrt.(jl[3]).* (2 ./jl[2]), 0.35*sqrt.(jl[3]).+sqrt.(jl[3]).*(2 ./jl[2]))))
save("$imgpath/c.png", map(clamp01nan,RGB.(0.2*sqrt.(jl[1] .* (2 ./jl[2]))*2,0.8*sqrt.(jl[1] .* (2 ./jl[2]))*2,0.3*sqrt.(jl[1] .* (2 ./jl[2]))*2 .+ jl[3]/2)))
save("$imgpath/d.png", map(clamp01nan,RGB.(sqrt.((2 ./jl[2]))*2 .* (jl[3]).^13 ./findmax((jl[3]).^13)[1], 0.1*(jl[3]).^(4).+0.45*jl[1].*sqrt.(2 ./jl[2]), 0.35*(jl[3]).^(4).+0.35*jl[1].*sqrt.(2 ./jl[2]))))
save("$imgpath/e.png", map(clamp01nan,2 .*RGB.(110*sqrt.(jl[1] ./jl[2] .* (1 .- jl[3]).^4).*sqrt.(shdw_mt),0.6.*sqrt.(jl[1] .* (2 ./jl[2])).*sqrt.(shdw_mt),0.7.*(jl[1] .* (2 ./jl[2])).^(1/3).*sqrt.(shdw_mt))))
save("$imgpath/f.png", map(clamp01nan,RGB.(sqrt.(jl[2] ./ findmax(jl[2])[1]).^(1/2) .* ((jl[3] ./ findmax(jl[3])[1]).^12),0.1 .*(jl[3]/findmax(jl[3])[1]).^6,(jl[2] ./ findmax(jl[2])[1]).^(1/2) .* (1 .-(jl[3] ./ findmax(jl[3])[1]).^12))))
save("$imgpath/g.png", map(clamp01nan,RGB.( jl[2]./maximum(jl[2]) ,(jl[1].*jl[3].^5 .- 1.5*(jl[5]./maximum(jl[5])).^2).*shdw_mt.+jl[2]./maximum(jl[2]),((jl[5]./maximum(jl[5])).^.8 .+ (1 .-(jl[1].*jl[3].^5))).*shdw_mt./2 .+jl[2]./maximum(jl[2]))))
#=
Notebook of cool c-Values:
- 0.517518,- 0.341729,- 0.407854,- 0.0716855
+ 0.415189, + 0.560351, + 0.174757, + 0.459138
+ 0.365658, - 0.0599171, - 0.396929, - 0.544048
- 0.254991, - 0.710382, - 0.110794, + 0.264363
- 0.250349, - 0.322733, - 0.667216, - 0.181973
+ 0.20848, + 0.0289175, - 0.989516, + 0.49807
- 0.214601, - 0.849091,+ 0.0252976,- 0.065572
- 0.577406,0.167387,- 0.482833,0.0
- 0.142171, + 0.231055, - 0.91313, - 0.0292274
- 0.234613, - 0.276556, - 0.876272, - 0.0177794
- 0.423564, - 0.691938, - 0.294925, + 0.0715491
=#

#### BONUS OPTIONS

canvas2=Array{RGB{Float64}}(undef,resy,resx)
ref=[RGB(158/255, 1/255, 66/255), RGB(215/255, 65/255, 78/255), RGB(237/255, 98/255, 70/255), RGB(249/255, 142/255, 82/255),RGB(253/255, 187/255, 108/255),RGB(254/255, 225/255, 141/255),RGB(1, 250/255, 182/255),RGB(239/255, 249/255, 166/255),RGB(205/255, 235/255, 157/255),RGB(177/255, 223/255, 163/255),RGB(148/255, 212/255, 164/255),RGB(126/255, 204/255, 165/255),RGB(102/255, 194/255, 165/255),RGB(84/255, 174/255, 173/255),RGB(68/255, 113/255, 178/255),RGB(91/255, 83/255, 164/255)]
#zmax2=round(log(findmax(zval_out2)[1]))/2
for (i,j) in ProgressBar(enumerate(jl[5])) #i=index, j=element
    j==0. && (canvas2[i]=RGB(0,0,0))
    j != 0. && (canvas2[i]=ref[Int(round(0.35*j+14)%16)+1])
end
#save("$imgpath/color_pass.png",canvas2.*jl[1])
#save("$imgpath/h.png",map(clamp01nan, canvas2.*(jl[1].*sqrt.(shdw_mt).+ jl[2]/findmax(jl[2])[1].+0.15*jl[3].^5)))

nrm_r=zeros(resy,resx)
nrm_g=zeros(resy,resx)
nrm_b=zeros(resy,resx)

Threads.@threads for i in ProgressBar(1:resx*resy)
    nrm_r[i]=nrm_mt[i][1]
    nrm_g[i]=nrm_mt[i][2]
    nrm_b[i]=nrm_mt[i][3]
end

#save("$imgpath/normals_pass.png",RGB.(map(clamp01nan,nrm_r),map(clamp01nan,nrm_g),map(clamp01nan,nrm_b)))


