# **UVCGAN v2**

## 1. abstract

1. An  unpaired image-to-image (**I2I**) translation technique seeks to find a mapping between two domains of data in a **fully unsupervised manner**.
2. **FID**（Fréchet Inception Distance）：一种用来评估生成模型质量的指标，它计算了生成图像和真实图像在一个预训练的模型（Inception v3）的特征空间中的距离，距离越小表示生成图像越接近真实图像。
3. **DMs**（diffusion models） suffer from limitations：
   1. not using data from the source domain during the training。
   2. maintaining consistency of the source and translated images only via simple pixel-wise errors。
4. **UVCGAN V2**是在**UCVGAN**的基础上做了改进的，并且证明了**pixel-wise I2I translation faithfulness metrics**的无效性，并建议对它们进行修改。



## 2. introduction

1. unpaired image-to-image translation常见的有三类模型：
   1. GANs：CycleGAN、STARGAN，SEAN，U-GAT-IT， etc.
   2. (variational) Autoencoder：这类模型通过一个encoder将输入图片压缩成一个隐向量，然后通过一个decoder将隐向量还原成输出图片。这类模型可以实现图片的Reconstruction、Denoising、Upsampling等功能。
   3. Generating flows：这类模型通过一个可逆的变换函数，将输入图片映射到一个简单的概率分布，然后通过逆变换函数将该分布的样本映射回图片空间。这类模型可以实现图片的生成、编辑、插值等功能。
2. **DMs**在unpaired image-to-image translation的应用：近期有一些研究利用DMs来实现无配对I2I转换，例如CycleNet3，UVC-GAN4和Denoising Diffusion Probabilistic Models for Text-Guided Image Manipulation。这些方法都试图通过不同的方式来保持源图像和目标图像的一致性，例如使用masked，attention，或者图像条件。
3. CycleGAN maintains an intrinsic consistency between the source and translated images(via the **cycle-consistency constraint**)—a feature that **cannot be achieved by simple pixel-wise consistency measures**.
4. Faithfulness and Realism:
   1.  Faithfulness captures the degree of **similarity** between the source and its translated image at an individual level.
   2. Realism attempts to estimate the **overlap** of the distributions of the translated images and the ones in the target.



## 3. contributions

1. introduce a novel **hybrid neural** architecture: ViT + Style Modulated Convolutional blocks.(利用了ViT的全局感知和风格调制卷积块的局部细节)
2. propose enhancing a common I2I **discriminator** architecture: a specialized batch head (prevent the problem of mode collapse )
3. incorporate modern GAN training techniques: exponential averaging of the generator weights、spectral normalization、unequal learning rates、improved zero-centered gradient penalty etc.
4.  extensive quantitative evaluations: FID，KID， etc.
5. 作者提出，当前unpaired image-to-image translation缺乏一个合适的**faithfulness metric**，这或许是**future work**里值得思考的idea。



