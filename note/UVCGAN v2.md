# **UVCGAN v2**

## 1. Abstract

1. An  unpaired image-to-image (**I2I**) translation technique seeks to find a mapping between two domains of data in a **fully unsupervised manner**.
2. **FID**（Fréchet Inception Distance）：一种用来评估生成模型质量的指标，它计算了生成图像和真实图像在一个预训练的模型（Inception v3）的特征空间中的距离，距离越小表示生成图像越接近真实图像。
3. **DMs**（diffusion models） suffer from limitations：
   1. not using data from the source domain during the training。
   2. maintaining consistency of the source and translated images only via simple pixel-wise errors。
4. **UVCGAN V2**是在**UCVGAN**的基础上做了改进的，并且证明了**pixel-wise I2I translation faithfulness metrics**的无效性，并建议对它们进行修改。



## 2. Introduction

1. unpaired image-to-image translation常见的有三类模型：
   1. **GANs**：CycleGAN、STARGAN，SEAN，U-GAT-IT， etc.
   2. **(variational) Autoencoder**：这类模型通过一个encoder将输入图片压缩成一个隐向量，然后通过一个decoder将隐向量还原成输出图片。这类模型可以实现图片的Reconstruction、Denoising、Upsampling等功能。
   3. **Generating flows**：这类模型通过一个可逆的变换函数，将输入图片映射到一个简单的概率分布，然后通过逆变换函数将该分布的样本映射回图片空间。这类模型可以实现图片的生成、编辑、插值等功能。
2. **DMs**在unpaired image-to-image translation的应用：近期有一些研究利用DMs来实现无配对I2I转换，例如CycleNet3，UVC-GAN4和Denoising Diffusion Probabilistic Models for Text-Guided Image Manipulation。这些方法都试图通过不同的方式来保持源图像和目标图像的一致性，例如使用masked，attention，或者图像条件。
3. CycleGAN maintains an intrinsic consistency between the source and translated images(via the **cycle-consistency constraint**)—a feature that **cannot be achieved by simple pixel-wise consistency measures**.
4. Faithfulness and Realism:
   1.  Faithfulness captures the degree of **similarity** between the source and its translated image at an individual level.
   2. Realism attempts to estimate the **overlap** of the distributions of the translated images and the ones in the target.



## 3. Contributions

1. introduce a novel **hybrid neural** architecture: ViT + Style Modulated Convolutional blocks.(利用了ViT的全局感知和风格调制卷积块的局部细节)
2. propose enhancing a common I2I **discriminator** architecture: a specialized batch head (prevent the problem of mode collapse )
3. incorporate modern GAN training techniques: exponential averaging of the generator weights、spectral normalization、unequal learning rates、improved zero-centered gradient penalty etc.
4.  extensive quantitative evaluations: FID，KID， etc.
5. 作者提出，当前unpaired image-to-image translation缺乏一个合适的**faithfulness metric**，这或许是**future work**里值得思考的idea。



## 4. Methods

### 4.1.  Review of the Original UVCGAN

UVCGAN follows the **CycleGAN** framework and uses a **hybrid UNet-ViT generator network**.

**UNet**是一种用于图像分割的卷积神经网络架构，其特点是具有**encoder**和**decoder**结构，**encoder**用于提取图像特征，**decoder**用于恢复图像的分辨率并生成分割结果。UNet通过将**encoder**和**decoder**连接起来，实现了从低分辨率到高分辨率的逐层上采样，从而提高了分割精度。UNet在医学图像分割领域取得了很好的效果，被广泛应用于各种医学图像分割任务。



### 4.2. Source-driven Style Modulation

![image-20240128185618702](C:\Users\clear\AppData\Roaming\Typora\typora-user-images\image-20240128185618702.png)

1. **架构设计**：UVCGANv2的**generator**被设计成能够推断每个输入图像的适当目标风格。这通过在生成器的解码分支中引入**style modulation**来实现，从而显著提高其表现力。
2. **bottleneck处理与style token**：在**generator**的**bottleneck**部分，图像被编码为一系列令牌，这些**token**被输入到**Transformer**网络中。这个**token**序列被增加一个额外的**learnable style token S**。**Transformer**输出处的**S token**状态作为潜在图像风格。
3. **style modulation**：对于U-Net解码分支中的每个卷积层，通过可训练的线性变换从S生成特定的风格向量$$S_i$$.
4. **效果**：通过这种方式，模型能够在生成过程中动态地调整其内部表示，以更好地匹配每个输入图像的风格。这不仅提高了生成图像的质量，还使得模型能够更灵活地处理各种输入和风格



### 4.3. Batch-Statistics-aware Discriminator

![image-20240128201428558](C:\Users\clear\AppData\Roaming\Typora\typora-user-images\image-20240128201428558.png)

由于**GANs**在训练时经常遭遇**mode collapse**问题，即**generator**倾向于生成相同或非常相似的样本，缺乏多样性。为了解决这个问题，其中一个解决方式就是使用**minibatch statistics**来增强**discriminator**的性能。

1. **minibatch statistics与内存效率**：传统的**minibatch statistics**需要较大的batch大小才能有效工作。然而，在处理高分辨率图像时，增加**batch size**会受到GPU内存的限制。为了解决这个问题，该模型提出了一种解耦**batch size**与**minibatch statistics**的方法，并设计了一种GPU内存高效的算法。
2. **使用cache**：该模型的主要思想是利用过去的**discriminator**特征缓存来替代大批量数据。这种缓存类似于“记忆库”（memory bank），在表示学习模型中已经被探索过，但在GANs中使用是新颖的。
3. **hybrid discrminator设计**：模型设计了一个由主体和**batch head**组成的**hybrid discrminator**。主体可以是任何常见的**discrminator**，而**batch head**则负责捕获批量统计信息，后跟两个卷积层（如图2所示）。这种模块化设计允许轻松替换不同的**discrminator**主体，同时保留**minibatch statistics**的能力。
4. **判别器批量头的输入**：判别器的批量头接收当前小批量的判别器主体输出与缓存中过去输出的历史记录的串联（沿批量维度）。这样，判别器在做出决策时可以同时考虑当前和过去的信息。



### 4.4. Pixel-wise Consistency Loss

为了提高生成图像和源图像之间的一致性，该模型在生成器损失中增加了一个额外的项$$L_{consist}$$。这个项捕获了源图像和翻译后图像在缩小尺寸后的$$L_1$$差异。具体来说，对于域A的图像，$$L_{consist,A}$$计算了从域A到域B的翻译图像$$G_{A→B(a)}$$和原始图像a在经过缩放函数F（将图像缩小到$$32×32$$像素）处理后的$$L_1$$差异。这个一致性损失项以$$λ_{consist}$$的权重加入到生成器损失中，适用于两个域。通过这种方式，模型在训练过程中会尽量保持生成图像和源图像在缩小尺寸后的一致性，从而提高了生成图像的质量。



### 4.5. Modern Training Techniques

1. 实现了**exponential averaging of the generator weights**，减少了**generator**对GAN训练中随机波动的依赖。
2. 实现了**spectral normalization of the discriminator weights**，增强了**discrminator**和整个训练过程的稳定性。
3. 尝试了使用**unequal learning rates for the generator and discriminator **，这在经验上已被证明可以提高模型性能。
4. 将通用的**gradient penalty**（GP）替换为改进的**zero-centered GP**，以促进GAN的收敛。这些现代训练技术的引入有助于提高模型的稳定性、收敛速度和生成图像的质量。

