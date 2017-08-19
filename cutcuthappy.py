import tensorflow as tf
#mnist =  # 读取图片数据集
sess = tf.InteractiveSession()# 创建session
# 一，函数声明部分

	def weight_variable(shape):
		# 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
		# ==================
		#这个和下面的一般用哪个 这个需要了解原理吗
        #tf.random_normal([3, 3, 1, 10]))
        # ==================
    		initial = tf.truncated_normal(shape, stddev=0.1)
    		return tf.Variable(initial)
	def bias_variable(shape):
		# 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    		initial = tf.constant(0.1, shape=shape)
    		return tf.Variable(initial)
	def conv2d(x, W):
        #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        #x表示输入的数据 图片的数量 高度 宽度 通道数
        #y表示卷积核大小
        #各个方向遍历补偿为1
        #padding越界填充方式
  		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	def max_pool_2x2(x):
        # tf.nn.max_pool( value, ksize,strides,padding,data_format=’NHWC’,name=None)
        # x输入的数据 图片的数量 高度 宽度 通道数
		# 池化的时候核心（面板）大小为2*2
        # 步数遍历的时候步数2，防止重叠
        # 如果越界周围补0，取最大值
  		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

# 二，定义输入输出结构

	# 声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
	xs = tf.placeholder(tf.float32, [None, 28*28])
	# 类别是0-9总共10个类别，对应输出分类结果
	ys = tf.placeholder(tf.float32, [None, 10])
    # x_image又把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
    x_image = tf.reshape(xs, [-1, 28, 28, 1])

# 三，搭建网络

    ## 卷积层 ##
	# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
	W_conv1 = weight_variable([5, 5, 1, 32])
	# 对于每一个卷积核都有一个对应的偏置量。
    #==================
    #偏置量b是凭经验吗？
    #==================
	b_conv1 = bias_variable([32])
	# 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
    #==================
    #这步是不是就是 W*X+b?
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #==================

	# 池化层 (28/2)*(28/2)*32
	h_pool1 = max_pool_2x2(h_conv1)


    ##全连接层 ##
	W_fc1 = weight_variable([14*14*32,64])
	b_fc1 = bias_variable([64])

	h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    ##全连接层 ##
	W_fc2 = weight_variable([14*14*32,64])
	b_fc2 = bias_variable([64])

	h_pool2_flat = tf.reshape(h_pool1, [-1, 14*14*32])

	h_fc2 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc2) + b_fc2)

