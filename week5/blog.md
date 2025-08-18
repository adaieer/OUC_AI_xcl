# 第五周 [生成对抗网络 & Diffussion](https://gitee.com/gaopursuit/ouc-dl/blob/master/week05.md)

## 1. 生成对抗网络 [参考文章](https://blog.csdn.net/anny_bili/article/details/100545297)
生成对抗网络 (GAN) 由两个模型构成, 生成模型 G 和判别模型 D, 随机噪声 z 通过 G 生成尽量服从真实数据分布 ${p_{data}}$ 的样本 G(z), 判别模型 D可以判断出输入样本是真实数据 x 还是生成数据G(z)。G 和 D 都可以是非线性的映射函数, 比如多层感知器。

训练判别器，损失函数为：
$\mathcal{L}_D = -\frac{1}{m}\sum_{i=1}^m [\log D(x_i) + \log(1 - D(G(z_i)))]$

训练生成器，损失函数为：
$\mathcal{L}_G = -\frac{1}{m}\sum_{i=1}^m \log D(G(z_i))$  

其中$D$为判别器，$G$为生成器，$z_i$为噪声，$x_i$为真实数据。

生成器与判别器交替训练，对于一个batch，训练一次判别器，优化后，固定判别器，再训练一次生成器，以此类推，直到所有batch训练完毕，即完成一个epoch。

## 2. GAN、cGAN、dcGAN 的区别

传统的GAN，其模型的训练是完全的无监督学习，无需考虑输入数据的标签，这存在模式崩溃（生成样本单一）的问题。于是就有了cGAN。

cGAN即条件生成对抗网络，在传统GAN的基础上，引入条件变量 y（如类别标签、文本描述）控制生成内容‌，实现有监督生成，可定向生成特定类别样本。

dcGAN即深度生成对抗网络，生成器与判别器均采用卷积神经网络（CNN）替代全连接层‌，网络结构增强，生成的样本更接近真实样本。

## 3. 代码运行结果
代码引用自 [github](https://github.com/wangguanan/Pytorch-Basic-GANs)，为适应本机环境，代码做了相应修改，限于时间即资源等因素，只跑了code/vanilla_gan.py，生成图片在gan_images文件夹下。
