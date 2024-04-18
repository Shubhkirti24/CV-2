#### Some decision made

1. Upsampling, downsampling: We will always upsample and downsample after applying the GaussianBlur, to speed up the computation by downsampling when we calculate the disparity.
We are using the off the shelf, PIL resize with interpolation method that is provided by them.

#### Disparity Map creation
Implemented according to the equation given in the assignment description. 

#### Naive Stereo. 

For Naive Stereo, after we simply create a 3D disparity map array, for each entry, we take the minimum energy level disparity. The disparity dimension and window size of the disparity map creation contributes to the smoothness of the depth map. Some images with different parameters are stored under this directory as experimental images.

<p align="center">
    <img src="https://github.iu.edu/cs-b657-sp2023/hirosato-nnimbale-ketvpate-shubpras-a2/blob/main/part3/maxdisparity10_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 10 and window size of 5</em>
</p>

<p align="center">
    <img src="https://github.iu.edu/cs-b657-sp2023/hirosato-nnimbale-ketvpate-shubpras-a2/blob/main/part3/Disparity20_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 20 and window size of 5</em>
</p>

<p align="center">
    <img src="https://github.iu.edu/cs-b657-sp2023/hirosato-nnimbale-ketvpate-shubpras-a2/blob/main/part3/Disparity20_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 20 and window size of 3</em>
</p>


