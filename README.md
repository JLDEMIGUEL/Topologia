# Topology API: Matrix and Simplicial Complex

This Topology API is a powerful tool for performing advanced mathematical computations involving matrix operations, simplicial complexes, alpha complexes, and Vietoris-Rips complexes. The API provides endpoints for calculating various matrix forms, as well as attributes and characteristics of different types of complexes used in topological analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [API Endpoints](#api-endpoints)

## Project Overview

This project implements a RESTful API for performing complex matrix operations and analyzing simplicial, alpha, and Vietoris-Rips complexes. The API is built using Flask, with extensive documentation provided by Flask-Smorest, and supports various mathematical computations through the use of libraries such as NumPy, SciPy, and SymPy. The project also integrates visualization tools to aid in the interpretation of computational results.

## Features

- **Matrix Operations**: Compute the Smith Normal Form of matrices over different rings such as integers (Z), rationals (Q), and within specific groups.
- **Simplicial Complex Analysis**: Determine various attributes of simplicial complexes, including dimension, Euler characteristic, Betti numbers, and more.
- **Alpha Complex Analysis**: Analyze alpha complexes derived from point clouds, providing insights into their topological structure.
- **Vietoris-Rips Complex Analysis**: Generate and analyze Vietoris-Rips complexes, including filtration orders, Betti numbers, and threshold values.
- **Topological Data Analysis**: Generate persistence diagrams and barcode diagrams for visualizing the topological features of data.
- **Dynamic Visualizations**: Create GIF animations representing the evolution of alpha complexes.
- **Testing**: A comprehensive suite of tests ensures the correctness and reliability of the code.

## Project Structure

```plaintext
├── app/
│   ├── alpha.py               # Handles Alpha complex calculations
│   ├── matrix.py              # Matrix operations related to the computations
│   ├── vietoris.py            # Handles Vietoris-Rips complex calculations
│   ├── ...
├── SimplicialComplex/
│   ├── AlphaComplex.py        # Implementation of AlphaComplex
│   ├── VietorisRipsComplex.py # Implementation of VietorisRipsComplex
│   ├── utilities.py           # Utility functions for complex computations
│   ├── tests/                 # Testing modules to validate the code
├── requirements.txt           # Project dependencies
```

## Technologies Used
- Programming Language: Python
- Frameworks & Libraries:
  - Flask: For building the REST API.
  - Flask-Smorest: Extends Flask with Swagger/OpenAPI documentation.
  - NumPy & SciPy: For numerical computations and linear algebra operations.
  - SymPy: Used for symbolic mathematics.
  - Matplotlib: Potentially used for visualizing computational results.
  - pytest & pytest-cov: For testing and coverage reporting.
  - imageio: Might be utilized for image data handling.

## API Endpoints

### Matrix Operations API: Smith Normal Form

#### Smith Normal Form in a Specific Group
- **URL**: `/matrix/smith_normal_form/<group>`
- **Method**: `POST`
- **Request Parameters**:
  - `group` (path parameter, required): A prime number representing the group for which the Smith normal form is calculated.
- **Request Body**:
  - `matrix` (list of lists of numbers): The matrix for which the Smith normal form is to be calculated.
- **Response**:
  - `matrix` (list of lists): The matrix in its Smith normal form.
  - `rows_opp_matrix` (list of lists): The row operations matrix.
  - `columns_opp_matrix` (list of lists): The column operations matrix.
  - `steps` (list): The steps taken to reach the Smith normal form, including intermediate matrices and descriptions.

#### Smith Normal Form with Coefficients in Q
- **URL**: `/matrix/smith_normal_form_q`
- **Method**: `POST`
- **Request Body**:
  - `matrix` (list of lists of fractions): The matrix with coefficients in the rational numbers (Q) for which the Smith normal form is to be calculated.
- **Response**:
  - `matrix` (list of lists): The matrix in its Smith normal form with coefficients as strings representing fractions.
  - `rows_opp_matrix` (list of lists): The row operations matrix with coefficients as strings representing fractions.
  - `columns_opp_matrix` (list of lists): The column operations matrix with coefficients as strings representing fractions.
  - `steps` (list): The steps taken to reach the Smith normal form, including intermediate matrices and descriptions.

#### Smith Normal Form with Coefficients in Z
- **URL**: `/matrix/smith_normal_form_z`
- **Method**: `POST`
- **Request Body**:
  - `matrix` (list of lists of integers): The matrix with coefficients in the integers (Z) for which the Smith normal form is to be calculated.
- **Response**:
  - `matrix` (list of lists): The matrix in its Smith normal form.
  - `rows_opp_matrix` (list of lists): The row operations matrix.
  - `columns_opp_matrix` (list of lists): The column operations matrix.
  - `steps` (list): The steps taken to reach the Smith normal form, including intermediate matrices and descriptions.

### Simplicial Complex API

#### Get All Complex Attributes
- **URL**: `/simplicial/all`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
  - Optional filters:
    - `n_faces_dim` (int): Dimension for which to calculate the number of faces.
    - `skeleton_dim` (int): Dimension for which to generate the skeleton.
    - `boundary_matrix_dim` (int): Dimension for which to generate the boundary matrix.
    - `star_face` (list): A face for which to calculate the star.
    - `closed_star_face` (list): A face for which to calculate the closed star.
    - `link_face` (list): A face for which to calculate the link.
- **Response**:
  - JSON containing various attributes such as faces list, dimension, Euler characteristic, connected components, Betti numbers, boundary matrices, and optionally calculated specific attributes like star, closed star, and link.

#### Get Complex Faces List
- **URL**: `/simplicial/faces_list`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `faces` (list): A list of all faces in the complex.

#### Get Complex Dimension
- **URL**: `/simplicial/dimension`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `dimension` (int): The dimension of the complex.

#### Get Number of Faces of a Specific Dimension
- **URL**: `/simplicial/n_faces/<int:dim>`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `n_faces` (list): A list of faces in the specified dimension.

#### Get Star of a Face
- **URL**: `/simplicial/star`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
  - `face` (list): The face for which to compute the star.
- **Response**:
  - `star` (list): The star of the selected face.

#### Get Closed Star of a Face
- **URL**: `/simplicial/closed_star`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
  - `face` (list): The face for which to compute the closed star.
- **Response**:
  - `closed_star` (list): The closed star of the selected face.

#### Get Link of a Face
- **URL**: `/simplicial/link`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
  - `face` (list): The face for which to compute the link.
- **Response**:
  - `link` (list): The link of the selected face.

#### Get Skeleton of a Specific Dimension
- **URL**: `/simplicial/skeleton/<int:dim>`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `skeleton` (list): The skeleton of the selected dimension.

#### Compute Euler Characteristic
- **URL**: `/simplicial/euler_characteristic`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `euler_characteristic` (int): The Euler characteristic of the complex.

#### Compute Connected Components
- **URL**: `/simplicial/connected_components`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `connected_components` (list): The connected components of the complex.

#### Get Boundary Matrix of a Specific Dimension
- **URL**: `/simplicial/boundary_matrix/<int:dim>`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `boundary_matrix` (list of lists): The boundary matrix for the selected dimension.

#### Get Generalized Boundary Matrix
- **URL**: `/simplicial/generalized_boundary_matrix`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `generalized_boundary_matrix` (list of lists): The generalized boundary matrix of the complex.

#### Compute Betti Number of a Specific Dimension
- **URL**: `/simplicial/betti_number/<int:dim>`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `betti_number` (int): The Betti number of the specified dimension.

#### Compute the First Two Betti Numbers
- **URL**: `/simplicial/incremental_algth`
- **Method**: `POST`
- **Request Body**:
  - `faces` (list of lists): A list of faces defining the simplicial complex.
- **Response**:
  - `bettis` (list of int): The first two Betti numbers of the complex.


### Alpha Complex API

#### Get Complete Complex Information
- **URL**: `/alpha/all`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
  - `n_faces_dim` (int, optional): Dimension of n-faces to retrieve.
  - `skeleton_dim` (int, optional): Dimension for the skeleton of the complex.
  - `boundary_matrix_dim` (int, optional): Dimension for the boundary matrix.
  - `star_face` (list of floats, optional): A face to compute the star of.
  - `closed_star_face` (list of floats, optional): A face to compute the closed star of.
  - `link_face` (list of floats, optional): A face to compute the link of.
- **Response**:
  - `faces`: Dictionary of faces with their corresponding thresholds.
  - `faces_list`: List of all faces.
  - `dimension`: Dimension of the Alpha Complex.
  - `euler`: Euler characteristic of the complex.
  - `components`: Connected components in the complex.
  - `bettis`: Betti numbers of the complex.
  - `general_boundary_matrix`: Generalized boundary matrix.
  - `filtration_order`: Filtration order of faces.
  - `threshold_values`: Threshold values for the complex.
  - `n_faces` (optional): Number of faces of the given dimension.
  - `skeleton` (optional): Skeleton of the complex.
  - `boundary_matrix` (optional): Boundary matrix of the given dimension.
  - `star` (optional): Star of the specified face.
  - `closed_star` (optional): Closed star of the specified face.
  - `link` (optional): Link of the specified face.

#### Get Faces Dictionary
- **URL**: `/alpha/faces`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - `faces`: Dictionary of faces with their corresponding thresholds.

#### Get Faces List
- **URL**: `/alpha/faces_list`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - `faces`: List of all faces.

#### Get Filtration Order
- **URL**: `/alpha/filtration_order`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - `faces`: Filtration order of faces.

#### Get Threshold Values
- **URL**: `/alpha/threshold_values`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - `threshold_values`: Threshold values for the complex.

#### Get Persistence Diagram
- **URL**: `/alpha/persistence_diagram`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - A `PNG` image of the persistence diagram.

#### Get Barcode Diagram
- **URL**: `/alpha/barcode_diagram`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - A `PNG` image of the barcode diagram.

#### Get Alpha Complex GIF
- **URL**: `/alpha/gif`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of list of floats): A list of points in n-dimensional space.
- **Response**:
  - A `GIF` animation representing the evolution of the Alpha Complex.

### Alpha Complex API

#### Get All Complex Attributes
- **URL**: `/vietoris/all`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
  - Optional filters:
    - `n_faces_dim` (int): Dimension for which to calculate the number of faces.
    - `skeleton_dim` (int): Dimension for which to generate the skeleton.
    - `boundary_matrix_dim` (int): Dimension for which to generate the boundary matrix.
    - `star_face` (list): A face for which to calculate the star.
    - `closed_star_face` (list): A face for which to calculate the closed star.
    - `link_face` (list): A face for which to calculate the link.
- **Response**:
  - JSON containing various attributes such as faces, faces list, dimension, Euler characteristic, connected components, Betti numbers, boundary matrices, filtration order, threshold values, and optionally calculated specific attributes like star, closed star, and link.

#### Get Complex Faces and Values
- **URL**: `/vietoris/faces`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - `faces` (dict): A dictionary where keys are faces and values are associated values.

#### Get Complex Faces List
- **URL**: `/vietoris/faces_list`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - `faces` (list): A list of all faces in the complex.

#### Get Filtration Order of Faces
- **URL**: `/vietoris/filtration_order`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - `faces` (list): A list of faces sorted by their associated float values.

#### Get Threshold Values of the Complex
- **URL**: `/vietoris/threshold_values`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - `threshold_values` (list): A list of threshold values in the complex.

#### Generate Persistence Diagram Image
- **URL**: `/vietoris/persistence_diagram`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - **Content-Type**: `image/png`
  - Binary image data representing the Persistence Diagram.

#### Generate Barcode Diagram Image
- **URL**: `/vietoris/barcode_diagram`
- **Method**: `POST`
- **Request Body**:
  - `points` (list of tuples): A list of points to generate the Vietoris-Rips complex.
- **Response**:
  - **Content-Type**: `image/png`
  - Binary image data representing the Barcode Diagram.