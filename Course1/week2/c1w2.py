import numpy as np

def swap_rows(M, row_index_1, row_index_2):
    """
    Swap rows in the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to perform row swaps on.
    - row_index_1 (int): Index of the first row to be swapped.
    - row_index_2 (int): Index of the second row to be swapped.
    """

    # Copy matrix M so the changes do not affect the original matrix. 
    M = M.copy()
    # Swap indexes
    M[[row_index_1, row_index_2]] = M[[row_index_2, row_index_1]]
    return M

def move_row_to_bottom(M, row_index):
    """
    Move the specified row to the bottom of the given matrix.

    Parameters:
    - M (numpy.array): Input matrix.
    - row_index (int): Index of the row to be moved to the bottom.

    Returns:
    - numpy.array: Matrix with the specified row moved to the bottom.
    """

    # Make a copy of M to avoid modifying the original matrix
    M = M.copy()

    # Extract the specified row
    row_to_move = M[row_index]

    # Delete the specified row from the matrix
    M = np.delete(M, row_index, axis=0)

    # Append the row at the bottom of the matrix
    M = np.vstack([M, row_to_move])

    return M

def get_index_first_non_zero_value_from_column(M, column, starting_row):
    """
    Retrieve the index of the first non-zero value in a specified column of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - column (int): The index of the column to search.
    - starting_row (int): The starting row index for the search.

    Returns:
    int or None: The index of the first non-zero value in the specified column, starting from the given row.
                Returns None if no non-zero value is found.
    """
    # Get the column array starting from the specified row
    column_array = M[starting_row:,column]
    for i, val in enumerate(column_array):
        # Iterate over every value in the column array. 
        # To check for non-zero values, you must always use np.isclose instead of doing "val == 0".
        if not np.isclose(val, 0, atol = 1e-5):
            # If one non zero value is found, then adjust the index to match the correct index in the matrix and return it.
            index = i + starting_row
            return index
    # If no non-zero value is found below it, return None.
    return None

def get_index_first_non_zero_value_from_row(M, row):
    """
    Find the index of the first non-zero value in the specified row of the given matrix.

    Parameters:
    - matrix (numpy.array): The input matrix to search for non-zero values.
    - row (int): The index of the row to search.

    Returns:
    int or None: The index of the first non-zero value in the specified row.
                Returns None if no non-zero value is found.
    """
    # Get the desired row
    row_array = M[row]
    for i, val in enumerate(row_array):
        # If finds a non zero value, returns the index. Otherwise returns None.
        if not np.isclose(val, 0, atol = 1e-5):
            return i
    return None

def augmented_matrix(A, B):
    """
    Create an augmented matrix by horizontally stacking two matrices A and B.

    Parameters:
    - A (numpy.array): First matrix.
    - B (numpy.array): Second matrix.

    Returns:
    - numpy.array: Augmented matrix obtained by horizontally stacking A and B.
    """
    augmented_M = np.hstack((A,B))
    return augmented_M



# GRADED FUNCTION: reduced_row_echelon_form
def reduced_row_echelon_form(A, B):
    """
    Utilizes elementary row operations to transform a given set of matrices, 
    which represent the coefficients and constant terms of a linear system, 
    into reduced row echelon form.

    Parameters:
    - A (numpy.array): The input square matrix of coefficients.
    - B (numpy.array): The input column matrix of constant terms

    Returns:
    numpy.array: A new augmented matrix in reduced row echelon form.
    """
    # Make copies of the input matrices to avoid modifying the originals
    A = A.copy()
    B = B.copy()


    # Convert matrices to float to prevent integer division
    A = A.astype('float64')
    B = B.astype('float64')

    # Number of rows in the coefficient matrix
    num_rows = len(A) 

    # List to store rows that should be moved to the bottom (rows of zeroes)
    rows_to_move = []

    ### START CODE HERE ###

    # Transform matrices A and B into the augmented matrix M
    M = np.hstack((A, B))
    
    # Iterate over the rows.
    for i in range(num_rows):

        # Find the first non-zero entry in the current row (pivot)
        pivot = M[i, i]
        # This variable stores the pivot's column index, it starts at i, but it may change if the pivot is not in the main diagonal.
        column_index = i


        # CASE PIVOT IS ZERO
        if np.isclose(pivot, 0): 
            # PART 1: Look for rows below current row to swap, you may use the function get_index_first_non_zero_value_from_column to find a row with non zero value
            index = get_index_first_non_zero_value_from_column(M, i, i)

            # If there is a non-zero pivot 
            if index is not None:
                # Swap rows if a non-zero entry is found below the current row
                M = swap_rows(M, i, index)

                # Update the pivot after swapping rows
                pivot = M[i, i]

            # PART 2 - NOT GRADED. This part deals with the case where the pivot isn't in the main diagonal.
            # If no non-zero entry is found below it to swap rows, then look for a non-zero pivot outside from diagonal.
            if index is None: 
                index_new_pivot = get_index_first_non_zero_value_from_row(M, i) 
                # If there is no non-zero pivot, it is a row with zeroes, save it into the list rows_to_move so you can move it to the bottom further.
                # The reason in not moving right away is that it would mess up the indexing in the for loop.
                # The second condition i >= num_rows is to avoid taking the augmented part into consideration.
                if index_new_pivot is None or index_new_pivot >= num_rows:
                    rows_to_move.append(i)
                    continue
                # If there is another non-zero value outside from diagonal, it will be the pivot.
                else:
                    pivot = M[i, index_new_pivot]
                    # Update the column index to agree with the new pivot position
                    column_index = index_new_pivot

        # END HANDLING FOR PIVOT 0   

            
        # Divide the current row by the pivot, so the new pivot will be 1. (reduced row echelon form)
        M[i] = M[i] / pivot

        # Perform row reduction for rows below the current row
        for j in range(i+1, num_rows):
            # Get the value in the row that is below the pivot value. Remember that the value in the column position is given by the variable called column_index
            value_below_pivot = M[j, column_index]
            
            # Perform row reduction using the formula:
            # row_to_reduce -> row_to_reduce - value_below_pivot * pivot_row
            M[j] = M[j] - value_below_pivot * M[i]
            
    ### END CODE HERE ###

    # Move every rows of zeroes to the bottom
    for row_index in rows_to_move:
        M = move_row_to_bottom(M,row_index)
    return M

# GRADED FUNCTION: check_solution
def check_solution(M):
    """
    Given an augmented matrix in reduced row echelon form, determine the nature of the associated linear system.

    Parameters:
    - M (numpy.array): An (n x n+1) matrix representing the augmented form of a linear system,
      where n is the number of equations and variables

    Returns:
    - str: A string indicating the nature of the linear system:
      - "Unique solution." if the system has one unique solution,
      - "No solution." if the system has no solution,
      - "Infinitely many solutions." if the system has infinitely many solutions.

    This function checks for singularity and analyzes the constant terms to determine the solution status.
    """
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows in the matrix
    num_rows = len(M)

    # Define the square matrix associated with the linear system
    coefficient_matrix = M[:,:-1]

    # Define the vector associated with the constant terms in the linear system
    constant_vector = M[:,-1]


    # Flag to indicate if the matrix is singular
    singular = False

    ### START CODE HERE ###

    # Iterate over the rows of the coefficient matrix
    for i in range(num_rows):

        # Test if the row from the square matrix has only zeros (do not replcae the part 'is None')
        if get_index_first_non_zero_value_from_row(coefficient_matrix, i) is None:
            # The matrix is singular, analyze the corresponding constant term to determine the type of solution
            singular = True 

            # If the constant term is non-zero, the system has no solution
            if not np.isclose(constant_vector[i], 0):
                return "No solution." 

    ### END CODE HERE ###

    # Determine the type of solution based on the singularity condition            
    if singular:        
        return "Infinitely many solutions."
    else:
        return "Unique solution."

# GRADED FUNCTION: back_substitution
def back_substitution(M):
    """
    Perform back substitution on an augmented matrix (with unique solution) in reduced row echelon form to find the solution to the linear system.

    Parameters:
    - M (numpy.array): The augmented matrix in reduced row echelon form (n x n+1).

    Returns:
    numpy.array: The solution vector of the linear system.
    """
    # Make a copy of the input matrix to avoid modifying the original
    M = M.copy()

    # Get the number of rows (and columns) in the matrix of coefficients
    num_rows = len(M)

    ### START CODE HERE ####
    
    # Iterate from bottom to top
    for i in range(num_rows - 1, 0, -1):
        # Get the substitution row
        substitution_row = M[i]

        # Iterate over the rows above the substitution_row
        for j in range(i - 1, -1, -1):
            # Get the row to be reduced
            row_to_reduce = M[j]

            # Get the index of the first non-zero element in the substitution row
            index = get_index_first_non_zero_value_from_row(M, i)

            # Get the value of the element at the found index
            value = row_to_reduce[index]


            # Perform the back substitution step using the formula row_to_reduce = None
            row_to_reduce = row_to_reduce - value * substitution_row

            # Replace the updated row in the matrix
            M[j] = row_to_reduce

    ### END CODE HERE ####

     # Extract the solution from the last column
    solution = M[:,-1]
    
    return solution

# GRADED FUNCTION: gaussian_elimination
def gaussian_elimination(A, B):
    """
    Solve a linear system represented by an augmented matrix using the Gaussian elimination method.

    Parameters:
    - A (numpy.array): Square matrix of size n x n representing the coefficients of the linear system
    - B (numpy.array): Column matrix of size 1 x n representing the constant terms.

    Returns:
    numpy.array or str: The solution vector if a unique solution exists, or a string indicating the type of solution.
    """

    ### START CODE HERE ###

    # Get the matrix in row echelon form
    reduced_row_echelon_M = reduced_row_echelon_form(A, B)

    # Check the type of solution (unique, infinitely many, or none)
    solution = check_solution(reduced_row_echelon_M)

    # If the solution is unique, perform back substitution
    if solution == "Unique solution.": 
        solution = back_substitution(reduced_row_echelon_M)
        
    ### END SOLUTION HERE ###

    return solution

from utils import string_to_augmented_matrix
equations = """
3*x + 6*y + 6*w + 8*z = 1
5*x + 3*y + 6*w = -10
4*y - 5*w + 8*z = 8
4*w + 8*z = 9
"""

variables, A, B = string_to_augmented_matrix(equations)

sols = gaussian_elimination(A, B)

if not isinstance(sols, str):
    for variable, solution in zip(variables.split(' '),sols):
        print(f"{variable} = {solution:.4f}")
else:
    print(sols)