
# Part 1

### Problem Description 

The aim of the given problem is to create a program that simulates the experience of taking off by a sequence of images formed by projecting a 3D scene into the 2D imaging plane of a camera, as the camera moves down the runway and increases its altitude. The projection matrixis used to project a 3D point onto a 2D plane. The program then uses projection matrix to create different views of the airport scene, as seen from a camera moving along a flight path.

The concepts involved in the problem are:

- 3D modeling: The airport scene is represented as a list of 3D points, which is used to create a 3D model of the airport.

- Projection: To create a 2D image of the 3D scene, each 3D point is encoded as a 4D homogeneous coordinate, multiplied by a 4x3 projection matrix Π, and then converted to a 2D point.

- Camera positioning and movement: The camera is positioned at a location in 3D space (tx, ty, tz) and has a certain orientation with tilt, twist, and yaw angles (α, β, and γ). The camera moves along a flight path, starting at the end of the runway, increasing altitude, banking right, flying parallel to the runway, and then landing.

- Animation: The program will generate a sequence of images by rendering the 3D scene from different camera positions as the camera moves along the flight path. The images will be displayed at a certain frame rate to create the illusion of motion. The animation will show the flight path of the camera as it takes off, banks, and lands, creating an oval-shaped flight path when viewed from above.


### Given data

We have been given a set of airport points that define a runway of length 1060 m and a control tower for the plane to fly around.

### Implementation :

#### Function Implementation:

1. **compute_homogeneous_coords**: 

The function takes a list of points as input and returns the same points with an additional fourth coordinate set to 1, which represents the homogeneous coordinate for transformations in 3D space.

2. **generate_projection_matrix**:

This function takes three arguments: the focal length of the camera, the position of the camera in 3D space, and the direction the camera is looking in terms of compass, tilt, and twist angles. The function returns a 3x4 projection matrix that can be used to project 3D points onto a 2D image plane. 

The function first calculates the rotation matrix from the camera's direction, then calculates the translation matrix from the camera position, and finally combines them with the intrinsic matrix to obtain the projection matrix.


3. **project_points**: 

This function takes a list of 3D points and a projection matrix as input and returns the projected 2D points on the image plane. 
The function first multiplies the points by the projection matrix, then applies the perspective division by dividing the resulting points by their z-coordinate. The function also excludes points with a z-coordinate of zero to avoid division by zero errors.

4. **clip_points**: 

This function takes a list of 2D points as input and returns only the points that have a non-negative z-coordinate. This is because points behind the camera are not visible in the image and should be excluded.

5. **draw_points**: 
This function takes a list of 2D points as input and draws the points on the axes. The function first draws lines between adjacent points, then draws circles at each point.


### Final Implementation :

The final implementation is done using the **animate_above** function, which takes a frame number as input and updates the camera position based on the frame number. It then generates a projection matrix using the generate_projection_matrix function, projects the points using the project_points function, clips the points using the clip_points function, and finally draws the clipped points on the axes using the draw_points function.




### Result:
Our implementation involved created the animation of take-offs and landings for the plane without any perspective change as  :

https://github.com/Shubhkirti24/CV-2/blob/main/part1/Bank_turn.gif

<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part1/takeoff.gif.gif" alt>
</p>
<p align="center">
     <em>Take off</em>
</p>


<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part1/Bank_turn.gif" alt>
</p>
<p align="center">
     <em>Bank Turn</em>
</p>



<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part1/landing.gif" alt>
</p>
<p align="center">
     <em>Landing</em>
</p>


---

# Part 2

### Problem Description :

Loopy Belief Propagation (LBP) is an algorithm that iteratively updates beliefs over a graphical model based on the beliefs of neighboring nodes. In this problem, the graphical model is a Markov Random Field, which encodes the relationships between neighboring houses. The goal was to find the assignment of political parties to houses that minimizes the cost function defined in the problem statement.

The first step is to represent the problem as a graphical model, where each node corresponds to a house and each edge corresponds to a fence between two neighboring houses. The value of each node is either R or D, indicating the political party to which the house is assigned. The value of each edge is either 0 or 1, indicating whether a fence is built between the two neighboring houses.

The cost function can be represented as a sum of local potentials and pairwise potentials over the nodes and edges of the graph, respectively. The local potential for each node  which depends on the contents of the R and D files. The pairwise potential for each edge is equal to 1000 if the two neighboring houses have different political parties and 0 otherwise.

### Given data

2 sample files which contains the grid containing the donation values and political affiliation.

### Implementation:

#### Function Implementation :

1. **create_matrix** : 

The function reads data from the file and returns a list of lists representing the data as a 2D matrix called 'map'.

2. **get_new_matrix** : 

The function creates a new matrix with dimensions specified by the argument and returns it as a list of dictionaries.

It then loops through each row and column of the matrix and initializes a dictionary for each cell, containing four entries 'L', 'R', 'U', and 'D'. Each entry itself is another dictionary containing two entries "RR" and "DD".

The 'L' entry corresponds to the left neighbor, 'R' entry corresponds to the right neighbor, 'U' entry corresponds to the top neighbor, and 'D' entry corresponds to the bottom neighbor. The "RR" entry denotes the minimum cost path to reach the cell from the source node with the Republican political party assignment, and the "DD" entry denotes the minimum cost path to reach the cell from the source node with the Democratic political party assignment.

3. **message_calc** :

The function calculates the message to be sent between two neighboring nodes.The inputs to the function are:

'ii': a tuple containing the row and column index of the current node
'jj': a tuple containing the row and column index of the neighboring node
'i_party': the political party assigned to the current node
'j_party': the political party assigned to the neighboring node
'sides': a tuple containing the number of rows and columns in the grid
'democrat_map': a dictionary containing the cost of assigning a house to the Democratic party
'republic_map': a dictionary containing the cost of assigning a house to the Republican party

The function first checks the political party assigned to the current node, and based on that, it gets the cost of assigning the neighboring node to the Democratic or Republican party from the respective dictionaries (democrat_map and republic_map).

If the political parties of the current and neighboring nodes are the same, the pairwise potential v_cost is set to 0. Otherwise, it is set to 1000.

Finally, the function calculates the final cost as the sum of the Democratic party assignment cost, the pairwise potential cost, and the previous message cost. The final cost is then returned by the function.

4. **old_message_calc** :

The function takes in the following parameters:

'i: an integer representing the row index of the current node
'j': an integer representing the column index of the current node
'sides': a string representing the side of the current node, which can be either "L", "R", "U", or "D"
'i_party': a string representing the political party of the current node, which can be either "RR" or "DD"
The function first initializes four variables L, R, U, and D to 0. These variables will be used to store the message values received from the neighboring nodes in the four directions.

The function then calculates the row and column indices of the neighboring nodes based on the current node's indices and side.

Next, the function checks if the neighboring nodes are within the bounds of the grid using the dim_s variable.

If a neighboring node is within bounds, the function retrieves the message value from that node's dictionary, which is stored in the old_matrix variable.

If a neighboring node is not within bounds, the message value is set to 0.

The function returns the sum of the message values received from the neighboring nodes.

5. **final_message_calc**:

The final_message function takes three arguments: ii, jj, and sides.

This function calculates the final message to be passed between two adjacent nodes, which represents the optimal message for the two parties involved in the communication.

Firstly, it creates an empty list d_v and an empty dictionary obj. It then loops through two different values of j_party (which represents the state of the party communicating with the current node), which can be either "RR" or "DD".

For each value of j_party, it calculates the minimum message value obtained for each of the two possible values of i_party, which can also be "RR" or "DD" using the **message_calc** function.
The resulting value is appended to the d_v list.

After computing the minimum value of d_v for the current value of j_party, it stores it in the obj dictionary with j_party as the key. The obj dictionary now stores the minimum value of the message that can be sent from the current node to the party in state "RR" or "DD".

6. **final_neighbour_cost** :

This function calculates the cost (sum of neighbour values) of a given cell at position (i,j) for a particular party (R or D).

If a neighbour exists, it retrieves the cost value of the current cell's neighbour cell for the given party. If the neighbour doesn't exist, the cost is considered as zero.

7. **final_cost** :

The function final_cost takes 'ii', which is a tuple containing the (row, column) index of the current cell in the grid, and 'i_party', which is the party affiliation of the current cell.

The function first checks the party affiliation of the current cell. If it is a Democrat (i_party is "DD"), the function looks up the cost of the Democrat in the democrat_map dictionary. If it is a Republican (i_party is "RR"), the function looks up the cost of the Republican in the republic_map dictionary.

The final_neighbour_cost function calculates the cost of the neighboring cells and returns the sum of the costs.

The function finally returns the sum of the cost of the current cell and the cost of its neighboring cells.


#### Final Implementation and Result :

1. The main loop of the program runs for n=500 iterations.

2. After the loop has finished, the old_matrix contains the minimum costs of each move for each grid cell. The code then initializes a new matrix new_matrix, where each cell is initially labeled as a Republican or Democratic district, depending on which party has the lower cost.

3. The last loop of the program iterates over each cell of the grid and updates its label based on the neighboring cells. Specifically, for each cell, the function final_cost is called twice, once for each party. The function returns the cost of assigning the current cell to that party, plus the costs of the neighboring cells in that party's district. The cell is then labeled with the party that has the lower cost.


### Result:

After the loop has finished, **new_matrix** contains the final redistricting map with the minimum cost.


### Alternate approach :

The alternate approach as found under 'test_2.py' was an unfinished approach based on libraries like 'networkx' used to create the grid. The approach was similar but complicated and inefficinent in its implementation to one chosen above.



---

# Part 3

## Problem Description :

The aim is to use Markov Random Fields (MRFs) to solve the stereo correspondence problem of creating a disparity map from a stereo pair of images. The stereo problem can be posed as an MRF inference problem, and a stereo energy function is defined using a unary cost function that estimates the likelihood of a pixel corresponding to a pixel in the other image, and a pairwise distance function. The implementation contains 2 parts : the Naive version, which should then be modified into a more general 'Loopy' version.


### Given data
We have been given 2 views of the objects with their respective Ground Truths.


### Implementation:

#### Function Implementation :

1. **mrf_stereo** :

The function takes in three arguments:

'img1': A numpy array representing the left image of the stereo pair.
'img2': A numpy array representing the right image of the stereo pair.
'disp_costs': A numpy array representing the cost function for the disparity values.

The function first creates a numpy array of zeros with the same shape as the left image, representing the disparity map. It then iterates over each pixel in the left image and assigns a random integer between 0 and MAX_DISPARITY as the disparity value for that pixel.

MAX_DISPARITY is a constant that represents the maximum possible disparity between the left and right images. The function then returns the randomly generated disparity map.

2. **disparity_costs**:

The given function computes the cost of disparity between two stereo images img1 and img2 using the Sum of Squared Differences (SSD) method. The cost is calculated for each pixel of img1 for a range of maximum disparity values (MAX_DISPARITY).

The function first initializes a 3D array result of shape (img1.shape[0], img1.shape[1], MAX_DISPARITY) to store the cost values. The window size W is set to 3, which defines the size of the square window used for computing the SSD.

The function then loops over each pixel of img1, excluding the border pixels within W pixels from the image boundary. Within the loop, the function computes the SSD cost for each disparity value by summing the squared differences of pixel intensities within the window centered at the current pixel in img1 and the corresponding window in img2, shifted by the disparity value. This is done by iterating over a range of window offsets Ws, and for each offset u and v, the pixel intensity difference between img1 and img2 at the current pixel and the corresponding pixel shifted by the disparity value is calculated, squared, and added the running sum.


3. **naive_stereo**:

The function applies np.argmin to disp_costs along the third axis, which corresponds to the disparity dimension. This returns an array that holds the index of the minimum cost for each pixel location in the image. Since the index corresponds to the disparity value, the resulting array represents the disparity map of the image.

4. For the "Loopy implementation" :

The same functions were optimised  and applied in the part 3 of the problem to add the Loopy approach to this problem.

#### Final implementation and Result :

The images are downsamples them by a factor of 2 to speed up the computation.We then checks if the ground truth image is provided and, it is scaled down by a factor of 3.

Next, disparity costs are computed. Thenstereo using both the naive and MRF techniques are calculated, respectively.

Finally, if a ground truth image is provided, the mean error of the two techniques is calculated using mean squared error (MSE).






### Result:
These are the output images (after going through the algorithm:

<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part3/maxdisparity10_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 10 and window size of 5</em>
</p>

<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part3/Disparity20_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 20 and window size of 5</em>
</p>

<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part3/Disparity20_window5.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 20 and window size of 3</em>
</p>

<p align="center">
    <img src="https://github.com/Shubhkirti24/CV-2/blob/main/part3/output-naive.png" alt>
</p>
<p align="center">
     <em>Naive Stereo Result with MAXDISPARITY of 40 and window size of 3</em>
</p>


### Observations:

- The window size determines the size of the neighborhood used for computing the matching cost. A larger window size can capture more contextual information and can be useful for handling occlusions and textureless regions, but it also increases the computational cost and can lead to inaccurate matches if the scene contains objects with small details or sharp edges.

- The maximum disparity value determines the range of possible disparities to consider. A larger maximum disparity value allows for a larger range of depth values to be estimated, but it also increases the computational cost and can lead to false matches if the images contain regions with large horizontal disparities, such as vertical edges or repeated patterns.


____


