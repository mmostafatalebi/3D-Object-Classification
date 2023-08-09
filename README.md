# 3D-Object-Classification

Abstract:

In this research paper, a novel technique is proposed for categorizing three-dimensional objects. The method involves utilizing a Voxel Grid VoxNet and Orion to accomplish the task. Instead  of  directly  analyzing  the  objects,  a  3D  voxel  grid representation  is  employed,  along  with  a  convolutional  neural network (CNN) for extracting features and making classifications. The  effectiveness  of  the  system  is  assessed  on  the  ModelNet-10 dataset, comparing the performance of VoxNet and Orion without considering  orientation.  The  paper  also  explores  the  impact  of these methods on network training and overall performance. The findings  reveal  how  different  combinations  of  techniques  can achieve  a  balance  between  accuracy,  efficiency,  and  preventing issues like over-fitting and under-fitting.

Introduction:

The field of deep learning is rapidly advancing in its ability to analyze  and  comprehend  three-dimensional  data.  As immersive technologies gain popularity, the significance of 3D data formats like meshes and point clouds continues to grow. While the real world exists in three dimensions, the amount of generated 3D data remains relatively limited compared to two- dimensional information.

In this project, we present my research on classifying 3D CAD models using a voxel grid neural network approach, evaluated on the widely-used Princeton ModelNet10 dataset. Voxel grid neural  networks  are  a  type  of  deep  learning  architecture designed for processing volumetric data represented as a three- dimensional  grid  of  voxels.  These  networks  offer  versatile applications, including object recognition, segmentation, and registration,  by  leveraging  the  inherent  3D  structure  of  the data.

In  alignment  with  the  concepts  described  in  the  article "Orientation-boosted Voxel Nets for 3D Object Recognition," We  employed  a  voxel  grid  approach  for  3D  object classification.  The  architecture  of  this  approach  is  based  on convolutional neural networks (CNNs) and operates on a 3D voxelized  input.  The  key  idea  is  to  voxelize  the  input  data, enabling  better  object  detection  in  the  three-dimensional space.

To enhance the classification model, We integrated ideas from both  VoxNet  and  Orion  methodologies.  We  increased  the number of hidden layers, similar to Orion, while incorporating data  augmentation  and  pooling  techniques.  The  resulting model  comprises  four  Convolutional  Neural  Networks  and two fully connected neural networks, each layer incorporating pooling operations, and augmented data to expand the dataset size.

Dataset:

In this project, we present our research on classifying 3D CAD models using a voxel grid neural network approach, evaluated on the widely-used Princeton ModelNet10 dataset.

Run:

Just run main.py file to test and see the results.
