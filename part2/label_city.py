import numpy as np

''' 
notes to self

- the file name  = cost to convert to a particular party
- create_matrx = opening file
- get_new_matrix = data structure
- message_calc  = U and V part calculation 
- old_message_calc = sum of all old messages calculation  


- we are storing the message at the source not the destination 
'''


def create_matrix(file_name):
    map = []
    with open(file_name, 'r') as file:
        for line in file:
            map.append([int(i) for i in line.split()])
    return map


def get_new_matrix(dis):

    li = np.empty(dis, dtype=dict)

    for i in range(dis[0]):
        for j in range(dis[1]):
            li[i, j] = {'L': {"RR": 0, "DD": 0}, 'R': {"RR": 0, "DD": 0}, 'U': {
                "RR": 0, "DD": 0}, 'D': {"RR": 0, "DD": 0}}

    return li


def message_calc(ii, jj, i_party, j_party, sides):

    # print(ii,jj)
    if i_party == "DD":

        d_cost = democrat_map[ii]

        if i_party == j_party:
            v_cost = 0
        else:
            v_cost = 1000

    elif i_party == "RR":

        d_cost = republic_map[ii]

        if i_party == j_party:
            v_cost = 0
        else:
            v_cost = 1000

    prev_cost = old_message_calc(ii[0], ii[1], sides, i_party)

    final_cost = d_cost + (v_cost) + prev_cost

    return final_cost


def old_message_calc(i, j, sides, i_party):

    L, R, U, D = 0, 0, 0, 0
    l_indx = (i, j-1)
    r_indx = (i, j+1)
    u_indx = (i+1, j)
    d_indx = (i-1, j)

    if sides == "L":

        # R
        if 0 <= r_indx[0] < dim_s[0] and 0 <= r_indx[1] < dim_s[1]:
            R = old_matrix[r_indx[0]][r_indx[1]][sides][i_party]
        else:
            R = 0
        # u
        if 0 <= u_indx[0] < dim_s[0] and 0 <= u_indx[1] < dim_s[1]:
            U = old_matrix[u_indx[0]][u_indx[1]][sides][i_party]
        else:
            U = 0
        # D
        if 0 <= d_indx[0] < dim_s[0] and 0 <= d_indx[1] < dim_s[1]:
            D = old_matrix[d_indx[0]][d_indx[1]][sides][i_party]
        else:
            D = 0

    elif sides == "R":

        # L
        if 0 <= l_indx[0] < dim_s[0] and 0 <= l_indx[1] < dim_s[1]:
            L = old_matrix[l_indx[0]][l_indx[1]][sides][i_party]
        else:
            L = 0

        # u
        if 0 <= u_indx[0] < dim_s[0] and 0 <= u_indx[1] < dim_s[1]:
            U = old_matrix[u_indx[0]][u_indx[1]][sides][i_party]
        else:
            U = 0

        # D
        if 0 <= d_indx[0] < dim_s[0] and 0 <= d_indx[1] < dim_s[1]:
            D = old_matrix[d_indx[0]][d_indx[1]][sides][i_party]
        else:
            D = 0

    elif sides == "U":

        # R
        if 0 <= r_indx[0] < dim_s[0] and 0 <= r_indx[1] < dim_s[1]:
            R = old_matrix[r_indx[0]][r_indx[1]][sides][i_party]
        else:
            R = 0

        # L
        if 0 <= l_indx[0] < dim_s[0] and 0 <= l_indx[1] < dim_s[1]:
            L = old_matrix[l_indx[0]][l_indx[1]][sides][i_party]
        else:
            L = 0

        # D
        if 0 <= d_indx[0] < dim_s[0] and 0 <= d_indx[1] < dim_s[1]:
            D = old_matrix[d_indx[0]][d_indx[1]][sides][i_party]
        else:
            D = 0

    elif sides == "D":

        # R
        if 0 <= r_indx[0] < dim_s[0] and 0 <= r_indx[1] < dim_s[1]:
            R = old_matrix[r_indx[0]][r_indx[1]][sides][i_party]
        else:
            R = 0

        # L
        if 0 <= l_indx[0] < dim_s[0] and 0 <= l_indx[1] < dim_s[1]:
            L = old_matrix[l_indx[0]][l_indx[1]][sides][i_party]
        else:
            L = 0

        # u
        if 0 <= u_indx[0] < dim_s[0] and 0 <= u_indx[1] < dim_s[1]:
            U = old_matrix[u_indx[0]][u_indx[1]][sides][i_party]
        else:
            U = 0

    return L+R+U+D


def final_message(ii, jj, sides):

    # print(ii,jj,sides)

    obj = {}
    d_v = []
    # for a value of J == state
    for j_party in ["RR", "DD"]:
        # minimize the value of the I == state
        for i_party in ["RR", "DD"]:

            d_v.append(message_calc(ii, jj, i_party, j_party, sides))

        obj[j_party] = min(d_v)



    return obj


def final_neighbour_cost(i, j, i_party):

    L, R, U, D = 0, 0, 0, 0
    l_indx = (i, j-1)
    r_indx = (i, j+1)
    u_indx = (i+1, j)
    d_indx = (i-1, j)
  # R
    if 0 <= r_indx[0] < dim_s[0] and 0 <= r_indx[1] < dim_s[1]:
        R = old_matrix[r_indx[0]][r_indx[1]]["R"][i_party]
    else:
        R = 0

        # L
    if 0 <= l_indx[0] < dim_s[0] and 0 <= l_indx[1] < dim_s[1]:
        L = old_matrix[l_indx[0]][l_indx[1]]["L"][i_party]
    else:
        L = 0

    # u
    if 0 <= u_indx[0] < dim_s[0] and 0 <= u_indx[1] < dim_s[1]:
        U = old_matrix[u_indx[0]][u_indx[1]]["U"][i_party]
    else:
        U = 0

    # D
    if 0 <= d_indx[0] < dim_s[0] and 0 <= d_indx[1] < dim_s[1]:
        D = old_matrix[d_indx[0]][d_indx[1]]["D"][i_party]
    else:
        D = 0

    return L+R+U+D


def final_cost(ii, i_party):

    if i_party == "DD":
        d_cost = democrat_map[ii]

    elif i_party == "RR":
        d_cost = republic_map[ii]

    prev_cost = final_neighbour_cost(ii[0], ii[1], i_party)

    return d_cost + prev_cost


if __name__ == "__main__":

    democrat_map = np.asarray(create_matrix("sample_d_bribes.txt"))
    republic_map = np.asarray(create_matrix("sample_r_bribes.txt"))

    dim_s = republic_map.shape

    old_matrix = get_new_matrix(dim_s)

    new_matrix = get_new_matrix(dim_s)

    n = 500

    while n != 0:
        for i in range(0, dim_s[0]):
            for j in range(0, dim_s[1]):

                for sides in ['U', 'D', 'L', 'R']:

                    ii = (i, j)
                    if sides == 'L':
                        jj = (i, j-1)

                    elif sides == 'R':
                        jj = (i, j+1)

                    elif sides == 'U':

                        jj = (i-1, j)

                    elif sides == 'D':

                        jj = (i+1, j)

                    if 0 <= jj[0] < dim_s[0] and 0 <= jj[1] < dim_s[1]:
                        # print(ii,jj)
                        new_matrix[i][j][sides] = final_message(
                            ii, jj, sides)  # {RR : 0 , DD : 0}
                        # print(obj)

        n -= 1
        old_matrix = new_matrix
        new_matrix = get_new_matrix(dim_s)
        ffinal_matrix = get_new_matrix(dim_s)



    for i in range(0, dim_s[0]):
        for j in range(0, dim_s[1]):

            obj = []

            for sides in ['RR', 'DD']:

                obj.append(final_cost((i, j), sides))
            # 00 = R and 11 = D
            dds = np.argmin(obj)
            if dds == 0:

                new_matrix[i][j] = "R"
                ffinal_matrix[i][j] = republic_map[i][j]
            
            elif dds == 1:    

                new_matrix[i][j] = "D" 
                ffinal_matrix[i][j] = democrat_map[i][j]



    total_cost = 0
    for i in range(0, dim_s[0]):
        for j in range(0, dim_s[1]):

            temp=0
            for sides in ['U', 'D', 'L', 'R']:

                    ii = (i, j)
                    if sides == 'L':
                        jj = (i, j-1)

                    elif sides == 'R':
                        jj = (i, j+1)

                    elif sides == 'U':

                        jj = (i-1, j)

                    elif sides == 'D':

                        jj = (i+1, j)

                    if 0 <= jj[0] < dim_s[0] and 0 <= jj[1] < dim_s[1]:
                        # temp += ffinal_matrix[ii]
                        if new_matrix[ii] != new_matrix[jj]:
                            temp += 500
            temp += ffinal_matrix[ii]            
            total_cost += temp

    print("\n",new_matrix,"\n")
    print(total_cost)


