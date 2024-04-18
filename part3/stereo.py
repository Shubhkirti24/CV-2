import random
import sys
import math
from PIL import Image, ImageFilter
import numpy as np
MAX_DISPARITY = 20  # Set this to the maximum disparity in the image pairs you'll use


def get_new_matrix(M, N, Disparity):
    li = np.empty((M, N), dtype=dict)
    for i in range(M):
      for j in range(N):
        li[i, j] = {"L": np.zeros((Disparity,)), "R": np.zeros((Disparity,)), "U": np.zeros((Disparity,)), "D": np.zeros((Disparity,))}

    return li


def old_message_calculation(i, j, sides, i_disparity):

    L, R, U, D = 0, 0, 0, 0
    l_index = (i, j-1)
    r_index = (i, j+1)
    u_index = (i+1, j)
    d_index = (i-1, j)
    if sides == "L":

        # R
        if 0 <= r_index [0] < image1.shape[0] and 0 <= r_index [1] < image1.shape[1]:
            R = old_matrix[r_index [0]][r_index [1]][sides]
        else:
            R = np.zeros((MAX_DISPARITY,))
        # u
        if 0 <= u_index [0] < image1.shape[0] and 0 <= u_index [1] < image1.shape[1]:
            U = old_matrix[u_index [0]][u_index [1]][sides]
        else:
            U = np.zeros((MAX_DISPARITY,))
        # D
        if 0 <= d_index [0] < image1.shape[0] and 0 <= d_index [1] < image1.shape[1]:
            D = old_matrix[d_index [0]][d_index [1]][sides]
        else:
            D = np.zeros((MAX_DISPARITY,))

    elif sides == "R":

        # L
        if 0 <= l_index [0] < image1.shape[0] and 0 <= l_index [1] < image1.shape[1]:
            L = old_matrix[l_index [0]][l_index [1]][sides]
        else:
            L = np.zeros((MAX_DISPARITY,))

        # u
        if 0 <= u_index [0] < image1.shape[0] and 0 <= u_index [1] < image1.shape[1]:
            U = old_matrix[u_index [0]][u_index [1]][sides]
        else:
            U = np.zeros((MAX_DISPARITY,))

        # D
        if 0 <= d_index [0] < image1.shape[0] and 0 <= d_index [1] < image1.shape[1]:
            D = old_matrix[d_index [0]][d_index [1]][sides]
        else:
            D = np.zeros((MAX_DISPARITY,))

    elif sides == "U":

         # R
        if 0 <= r_index [0] < image1.shape[0] and 0 <= r_index [1] < image1.shape[1]:
            R = old_matrix[r_index [0]][r_index [1]][sides]
        else:
            R = np.zeros((MAX_DISPARITY,))

        # L
        if 0 <= l_index [0] < image1.shape[0] and 0 <= l_index [1] < image1.shape[1]:
            L = old_matrix[l_index [0]][l_index [1]][sides]
        else:
            L = np.zeros((MAX_DISPARITY,))

        # D
        if 0 <= d_index [0] < image1.shape[0] and 0 <= d_index [1] < image1.shape[1]:
            D = old_matrix[d_index [0]][d_index [1]][sides]
        else:
            D = np.zeros((MAX_DISPARITY,))

    elif sides == "D":

                 # R
        if 0 <= r_index [0] < image1.shape[0] and 0 <= r_index [1] < image1.shape[1]:
            R = old_matrix[r_index [0]][r_index [1]][sides]
        else:
            R = np.zeros((MAX_DISPARITY,))

        # L
        if 0 <= l_index [0] < image1.shape[0] and 0 <= l_index [1] < image1.shape[1]:
            L = old_matrix[l_index [0]][l_index [1]][sides]
        else:
            L = np.zeros((MAX_DISPARITY,))

        # u
        if 0 <= u_index [0] < image1.shape[0] and 0 <= u_index [1] < image1.shape[1]:
            U = old_matrix[u_index [0]][u_index [1]][sides]
        else:
            U = np.zeros((MAX_DISPARITY,))
    return L+R+U+D


def message_calculation(ii, jj, i_disparity, sides, disp_costs,prev_cost,dsi_indx):
  message = []
  neighbor_disparity = np.arange(MAX_DISPARITY)
  smoothness_cost = np.square(neighbor_disparity - i_disparity)
  #for neighbor_disparity in range(j_disparity):
  data_cost = disp_costs[ii]
  #print(f"Shape of data_cost {data_cost.shape}")
  #print(f"Shape of smoothness {smoothness_cost.shape}")

  
    #smoothness_cost = (neighbor_disparity - i_disparity)**2
  
  #print(f"Shape of prev_cost {prev_cost.shape}")

  #message.append(prev_cost + data_cost + 0.8 * smoothness_cost)
    # find minimum
    # message.appen(np.minimum(temp))
  #print(f"Smoothness costs {smoothness_cost}")
  return min(prev_cost[dsi_indx] + data_cost +1.5 * smoothness_cost)


def final_message(ii, jj, sides, disp_costs):

  objective = {}
  # for j_disparity in range(MAX_DISPARITY):
  for dsi_indx,i_disparity in enumerate(range(MAX_DISPARITY)):
      #total_costs.append(message_calculation(ii, jj, i_disparity, sides, disp_costs))
      prev_cost = old_message_calculation(ii[0], ii[1], sides, i_disparity)
      print(prev_cost)
      objective[i_disparity] = message_calculation(ii, jj, i_disparity, sides, disp_costs,prev_cost,dsi_indx)
  #print(objective)
  return objective


def final_cost(ii, i_disparity, disparity_costs):

  # I only have one datacost
  d_cost = disparity_costs[ii][i_disparity]
    # for d in range(MAX_DISPARITY):
    #   d_cost = disparity_costs[ii][d]
    # if i_disparity == "DD":
    #     d_cost = democrat_map[ii]

    # elif i_disparity == "RR":
    #     d_cost = republic_map[ii]

  prev_cost = final_neighbour_cost(ii[0], ii[1], i_disparity)

  return d_cost + prev_cost


def final_neighbour_cost(i, j, i_disparity):
        img_dim = original_image_size
        L, R, U, D = 0, 0, 0, 0
        l_index  = (i, j-1)
        r_index  = (i, j+1)
        u_index  = (i+1, j)
        d_index  = (i-1, j)
      # R
        if 0 <= r_index [0] < image1.shape[0] and 0 <= r_index [1] <image1.shape[1]:
            R = old_matrix[r_index [0]][r_index [1]]["R"][i_disparity]
        else:
            R = 0

         # L
        if 0 <= l_index [0] < image1.shape[0] and 0 <= l_index [1] <image1.shape[1]:
            L = old_matrix[l_index [0]][l_index [1]]["L"][i_disparity]
        else:
            L = 0

        # u
        if 0 <= u_index [0] < image1.shape[0] and 0 <= u_index [1] <image1.shape[1]:
            U = old_matrix[u_index [0]][u_index [1]]["U"][i_disparity]
        else:
            U = 0

        # D
        if 0 <= d_index [0] < image1.shape[0] and 0 <= d_index [1] <image1.shape[1]:
            D = old_matrix[d_index [0]][d_index [1]]["D"][i_disparity]
        else:
            D = 0

        return L+R+U+D


def mrf_stereo(image1, image2, disp_costs):
    # this placeholder just returns a random disparity map
    result = np.zeros((image1.shape[0], image1.shape[1]))
    #epoch = int(np.sqrt((image1.shape[0])**2 + (image1.shape[1])**2))
    new_matrix=get_new_matrix(original_image_size[0],original_image_size[1], MAX_DISPARITY)
    epoch = 20
    while epoch != 0:
      for i in range(image1.shape[0]):
          for j in range(image1.shape[1]):
            for sides in ['L', 'R', 'U', 'D']:
              index=(i, j)
              # we get the neighbors.
              if (sides == 'L'):
                jj=(i, j-1)
              elif (sides == 'R'):
                jj=(i, j+1)
              elif (sides == 'U'):
                jj=(i-1, j)
              elif (sides == 'D'):
                jj=(i+1, j)

              if (0 <= jj[0] < image1.shape[0] and 0 <= jj[1] < image1.shape[1]):
                new_matrix[i][j][sides]=final_message(index, jj, sides, disp_costs)
      epoch -= 1
      break
      print(f"Epoch: {epoch}")
      old_matrix=new_matrix
      new_matrix=get_new_matrix(original_image_size[0],original_image_size[1], MAX_DISPARITY)

    print(old_matrix)

    print("final calculation started")
      # result = np.empty(image1.shape)
    for i in range(0, image1.shape[0]):
      for j in range(0, image1.shape[1]):
        obj=[]
        for disparity in range(MAX_DISPARITY):
        # need to figure this part out.
                    # for sides in ['RR', 'DD']:
          obj.append(final_cost((i, j), disparity, disp_costs))
      # 00 = R and 11 = D
          result[i][j]=np.argmin(obj)
        # new_matrix[i][j] = np.argmin(obj)

            # result[i,j] = random.randint(0, MAX_DISPARITY)
    return result


def disparity_costs(image1, image2):
  # result holds a cost.
  result=np.ones((image1.shape[0], image1.shape[1], MAX_DISPARITY),dtype="float64") * 10**(6)
  print(result)
  #return result
  W=3
  # half_kernel = int(W/2)
  height=image1.shape[0]
  width=image1.shape[1]
  # print(image1.shape)
  # print(image2.shape)
  Ws=np.arange(-W, W)

  for j in range(W, (height-W)):
    for i in range(W, (width-W)):
      # sum = 0
      for d in range(MAX_DISPARITY):
        sum=0
        for u in Ws:
          for v in Ws:
            diff=image1[j+v][i+u] - image2[j+v][i+u-d]
            diff=diff**2
            #print(diff)
            sum += diff
        result[j][i][d]=sum
  #print(result)
  return result


# # This function should compute the function D() in the assignment
# def disparity_costs(image1, image2):
#     # this placeholder just returns a random cost map

#     result = np.zeros((image1.shape[0], image1.shape[1], MAX_DISPARITY))
#     for i in range(image1.shape[0]):
#         for j in range(image1.shape[1]):
#             for d in range(MAX_DISPARITY):
#                 result[i,j,d] = random.randint(0, 255)
#     return result

# This function finds the minimum cost at each pixel
def naive_stereo(image1, image2, disp_costs):
    return np.argmin(disp_costs, axis=2)

if __name__ == "__main__":
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        raise Exception(
            "usage: " + sys.argv[0] + " image_file1 image_file2 [gt_file]")
    input_filename1, input_filename2=sys.argv[1], sys.argv[2]

    # read in images and gt
    image1 = Image.open(input_filename1).convert("L")
    image2 = Image.open(input_filename2).convert("L")
    image1=np.array(image1,dtype='int64')
    image2=np.array(image2,dtype='int64')

    # saving original image size for later upsampling. (width, height)
    original_image_size=(image1.shape[1], image1.shape[0])
    # downsampling the image to speed up the computation.
    image1=image1[::2, ::2]/255
    image2=image2[::2, ::2]/255
    #image1=image1[::4, ::4] / 255
    #image2=image2[::4, ::4] / 255
    gt=None
    if len(sys.argv) == 4:
        gt=np.array(Image.open(sys.argv[3]))[:, :, 0]

        # gt maps are scaled by a factor of 3, undo this...
        gt=gt / 3.0

    # compute the disparity costs (function D_2())
    #disp_costs=disparity_costs(image1, image2)
    d_map = disparity_costs(image1, image2)
    print("Disparity map created")
    #print(np.unique(disp_costs))
    print(d_map)
    # do stereo using naive technique
    disp1=naive_stereo(image1, image2, d_map)
    #print(disp1,type(disp1))
    disp1_=Image.fromarray(disp1.astype(np.uint8))
    disp1_resized=disp1_.resize(original_image_size, Image.LANCZOS)
    disp1_resized.save("output-naive.png")
    # Image.fromarray(disp1.astype(np.uint8)).save("output-naive.png")
    print("Naive completed")
    # do stereo using mrf
    old_matrix=get_new_matrix(image1.shape[0], image1.shape[1], MAX_DISPARITY)
    # new_matrix=get_new_matrix(image1.shape[0], image1.shape[1], MAX_DISPARITY)
    disp3=mrf_stereo(image1, image2, d_map)
    print("mrf completed")
    disp3_=Image.fromarray(disp3.astype(np.uint8))
    disp3_resized=disp3_.resize(original_image_size, Image.LANCZOS)
    disp3_resized.save("output-mrf.png")

    # Measure error with respect to ground truth, if we have it...
    if gt is not None:
        err=np.sum((disp1 - gt)**2)/gt.shape[0]/gt.shape[1]
        print("Naive stereo technique mean error = " + str(err))

        err=np.sum((disp3 - gt)**2)/gt.shape[0]/gt.shape[1]
        print("MRF stereo technique mean error = " + str(err))
