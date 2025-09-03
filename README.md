# PET/CT project
Computer-aided diagnosis (CAD) systems are essential in assisting radiologists diagnose lung cancer more quickly, and accurately, with fewer diagnostic errors. the aim of this study is To automatically diagnose lung cancer from 2D (18F) fluorodeoxyglucose (FDG) PET/CT images using convolutional neural networks (CNNs).

This study which was approved by Shiraz University of Medical Sciences Institutional Review Board (IRB) (IR.SUMS.REC.1401.715) included patients who were recruited from Kowsar charity hospital, in Shiraz, Iran. 

The convolutional neural network (CNN) architecture is the most effective algorithm for improving deep learning in computer vision. In this study, some of the most popular and well-designed deep convolutional networks were utilized. Also, transfer learning is the primary technique employed, enabling the networks to benefit from pre-learned knowledge rather than training them from scratch. Figure1 shows the convolutional neural network design applied.

![Screenshot 2023-02-22 232056](https://github.com/MKarimpour1997/thesProjectPET-CT/assets/131992544/9db90fe8-435b-4dff-addc-e000a7f50532)
Figure 1. The convolutional neural network design for lung cancer diagnosis


Convolutional neural networks have the capability of not only classifying images but also localizing class-specific regions within the image. Class activation mapping (CAM) is a technique that enables us to identify the parts of the image that have contributed the most to the CNN model’s output. In the context of localizing lung cancer, the CNN model architecture used in this study deviated from the conventional approach employed for image classification (Figure 1). Instead of incorporating multiple fully connected dense layers at the end of the CNN model, a global average pooling layer was added immediately after the last layer of the pre-trained CNN network. This choice was made to preserve the spatial information in the output of the last convolutional layer. Finally, an output softmax layer with two neurons was added to classify images (Figure 2).

![Class activation map figure](https://github.com/MKarimpour1997/thesProjectPET-CT/assets/131992544/048fecd6-90b2-446e-877c-30cdd24f4613)
Figure 2. The convolutional neural network design for lung cancer localization


For the Custom Res-SE Net scripts, see this repository: https://github.com/nedatghd/Lung-Cancer-Diagnosis-from-2D-18F---PET-CT-Images
