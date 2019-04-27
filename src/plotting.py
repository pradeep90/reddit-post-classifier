import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PLOT_NBC = False
PLOT_LR = True
PLOT_CNN = False

# Learing curve for NBC
if PLOT_NBC:
	training_fracs = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 1.0]
	training_precision_at_5 = [1, 0.9999166666666667, 0.9998333333333334, 0.9995833333333334, 0.9993981481481482, 0.99925, 0.9919013888888889]
	testing_precision_at_5 = [0.58738, 0.72416, 0.77479, 0.80193, 0.83226, 0.84762, 0.89345]

	plot_name = 'learning_curve_nbc_{}.png'.format(str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(training_fracs, training_precision_at_5, label='train')
	line2, = ax.plot(training_fracs, testing_precision_at_5, label='test')
	ax.legend()
	title='Training and Test Accuracies for v.s. Training Fraction'
	plt.xlabel('training_fracs')
	plt.ylabel('Precision@5')
	plt.title(title)
	plt.savefig(plot_name)

if PLOT_LR:
	training_fracs = [0.025, 0.05, 0.075, 0.1, 0.15]
	training_precision_at_5 = [1, 1, 1, 1, 1]
	testing_precision_at_5 = [0.77, 0.82, 0.84, 0.85, 0.86]

	plot_name = 'learning_curve_lr_{}.png'.format(str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(training_fracs, training_precision_at_5, label='train')
	line2, = ax.plot(training_fracs, testing_precision_at_5, label='test')
	ax.legend()
	title='Training and Test Accuracies for v.s. Training Fraction'
	plt.xlabel('training_fracs')
	plt.ylabel('Precision@5')
	plt.title(title)
	plt.savefig(plot_name)


if PLOT_CNN:

	data = """
	80000/80000 [==============================] - 221s 3ms/step - loss: 6.6233 - acc: 0.0040 - top_k_categorical_accuracy: 0.0174 - val_loss: 6.0892 - val_acc: 0.0098 - val_top_k_categorical_accuracy: 0.0420

	80000/80000 [==============================] - 221s 3ms/step - loss: 5.5705 - acc: 0.0277 - top_k_categorical_accuracy: 0.0998 - val_loss: 5.2726 - val_acc: 0.0420 - val_top_k_categorical_accuracy: 0.1362
	

	80000/80000 [==============================] - 223s 3ms/step - loss: 4.9008 - acc: 0.0701 - top_k_categorical_accuracy: 0.2075 - val_loss: 4.7563 - val_acc: 0.0891 - val_top_k_categorical_accuracy: 0.2454
	

	80000/80000 [==============================] - 225s 3ms/step - loss: 4.4173 - acc: 0.1218 - top_k_categorical_accuracy: 0.3048 - val_loss: 4.5053 - val_acc: 0.1213 - val_top_k_categorical_accuracy: 0.3004
	

	80000/80000 [==============================] - 224s 3ms/step - loss: 4.0409 - acc: 0.1718 - top_k_categorical_accuracy: 0.3845 - val_loss: 4.3248 - val_acc: 0.1529 - val_top_k_categorical_accuracy: 0.3457
	

	80000/80000 [==============================] - 222s 3ms/step - loss: 3.7427 - acc: 0.2126 - top_k_categorical_accuracy: 0.4449 - val_loss: 4.2171 - val_acc: 0.1756 - val_top_k_categorical_accuracy: 0.3790
	

	80000/80000 [==============================] - 221s 3ms/step - loss: 3.4914 - acc: 0.2529 - top_k_categorical_accuracy: 0.4950 - val_loss: 4.3441 - val_acc: 0.1744 - val_top_k_categorical_accuracy: 0.3719
	

	80000/80000 [==============================] - 221s 3ms/step - loss: 3.2783 - acc: 0.2870 - top_k_categorical_accuracy: 0.5333 - val_loss: 4.1594 - val_acc: 0.1989 - val_top_k_categorical_accuracy: 0.4041
	

	80000/80000 [==============================] - 220s 3ms/step - loss: 3.0868 - acc: 0.3154 - top_k_categorical_accuracy: 0.5703 - val_loss: 4.2744 - val_acc: 0.1933 - val_top_k_categorical_accuracy: 0.3917
	

	80000/80000 [==============================] - 220s 3ms/step - loss: 2.9134 - acc: 0.3444 - top_k_categorical_accuracy: 0.6034 - val_loss: 4.2295 - val_acc: 0.2097 - val_top_k_categorical_accuracy: 0.4143
	

	80000/80000 [==============================] - 222s 3ms/step - loss: 2.7519 - acc: 0.3720 - top_k_categorical_accuracy: 0.6330 - val_loss: 4.3870 - val_acc: 0.2099 - val_top_k_categorical_accuracy: 0.4190
	

	80000/80000 [==============================] - 222s 3ms/step - loss: 2.6156 - acc: 0.3946 - top_k_categorical_accuracy: 0.6581 - val_loss: 4.4451 - val_acc: 0.2042 - val_top_k_categorical_accuracy: 0.4023
	

	80000/80000 [==============================] - 233s 3ms/step - loss: 2.4792 - acc: 0.4192 - top_k_categorical_accuracy: 0.6820 - val_loss: 4.5719 - val_acc: 0.2061 - val_top_k_categorical_accuracy: 0.4101
	

	80000/80000 [==============================] - 227s 3ms/step - loss: 2.3596 - acc: 0.4394 - top_k_categorical_accuracy: 0.7028 - val_loss: 4.7459 - val_acc: 0.1976 - val_top_k_categorical_accuracy: 0.4000
	

	80000/80000 [==============================] - 243s 3ms/step - loss: 2.2500 - acc: 0.4592 - top_k_categorical_accuracy: 0.7230 - val_loss: 4.8310 - val_acc: 0.2021 - val_top_k_categorical_accuracy: 0.4004
	

	80000/80000 [==============================] - 242s 3ms/step - loss: 2.1556 - acc: 0.4772 - top_k_categorical_accuracy: 0.7412 - val_loss: 4.9553 - val_acc: 0.2046 - val_top_k_categorical_accuracy: 0.4051
	

	80000/80000 [==============================] - 243s 3ms/step - loss: 2.0611 - acc: 0.4944 - top_k_categorical_accuracy: 0.7579 - val_loss: 5.1070 - val_acc: 0.1941 - val_top_k_categorical_accuracy: 0.3936
	

	80000/80000 [==============================] - 242s 3ms/step - loss: 1.9717 - acc: 0.5104 - top_k_categorical_accuracy: 0.7723 - val_loss: 5.3036 - val_acc: 0.2042 - val_top_k_categorical_accuracy: 0.4021
	

	80000/80000 [==============================] - 240s 3ms/step - loss: 1.8947 - acc: 0.5250 - top_k_categorical_accuracy: 0.7874 - val_loss: 5.4930 - val_acc: 0.1991 - val_top_k_categorical_accuracy: 0.3966
	

	80000/80000 [==============================] - 238s 3ms/step - loss: 1.8224 - acc: 0.5395 - top_k_categorical_accuracy: 0.7985 - val_loss: 5.7421 - val_acc: 0.1953 - val_top_k_categorical_accuracy: 0.3928
	

	80000/80000 [==============================] - 237s 3ms/step - loss: 1.7553 - acc: 0.5534 - top_k_categorical_accuracy: 0.8112 - val_loss: 5.7278 - val_acc: 0.1931 - val_top_k_categorical_accuracy: 0.3948
	

	80000/80000 [==============================] - 236s 3ms/step - loss: 1.6928 - acc: 0.5660 - top_k_categorical_accuracy: 0.8206 - val_loss: 5.8661 - val_acc: 0.1908 - val_top_k_categorical_accuracy: 0.3825
	

	80000/80000 [==============================] - 240s 3ms/step - loss: 1.6367 - acc: 0.5775 - top_k_categorical_accuracy: 0.8308 - val_loss: 6.0282 - val_acc: 0.1882 - val_top_k_categorical_accuracy: 0.3840
	

	80000/80000 [==============================] - 237s 3ms/step - loss: 1.5811 - acc: 0.5891 - top_k_categorical_accuracy: 0.8403 - val_loss: 6.3243 - val_acc: 0.1915 - val_top_k_categorical_accuracy: 0.3886
	

	80000/80000 [==============================] - 235s 3ms/step - loss: 1.5333 - acc: 0.5996 - top_k_categorical_accuracy: 0.8481 - val_loss: 6.1779 - val_acc: 0.1834 - val_top_k_categorical_accuracy: 0.3750
	

	80000/80000 [==============================] - 255s 3ms/step - loss: 1.4889 - acc: 0.6095 - top_k_categorical_accuracy: 0.8551 - val_loss: 6.5050 - val_acc: 0.1883 - val_top_k_categorical_accuracy: 0.3822
	

	80000/80000 [==============================] - 266s 3ms/step - loss: 1.4466 - acc: 0.6185 - top_k_categorical_accuracy: 0.8629 - val_loss: 6.6663 - val_acc: 0.1800 - val_top_k_categorical_accuracy: 0.3674
	

	80000/80000 [==============================] - 259s 3ms/step - loss: 1.4101 - acc: 0.6263 - top_k_categorical_accuracy: 0.8694 - val_loss: 6.7527 - val_acc: 0.1837 - val_top_k_categorical_accuracy: 0.3735
	

	80000/80000 [==============================] - 272s 3ms/step - loss: 1.3773 - acc: 0.6326 - top_k_categorical_accuracy: 0.8748 - val_loss: 6.9058 - val_acc: 0.1862 - val_top_k_categorical_accuracy: 0.3782
	

	80000/80000 [==============================] - 253s 3ms/step - loss: 1.3401 - acc: 0.6404 - top_k_categorical_accuracy: 0.8797 - val_loss: 7.1382 - val_acc: 0.1855 - val_top_k_categorical_accuracy: 0.3767
	

	80000/80000 [==============================] - 247s 3ms/step - loss: 1.3198 - acc: 0.6465 - top_k_categorical_accuracy: 0.8853 - val_loss: 7.1970 - val_acc: 0.1823 - val_top_k_categorical_accuracy: 0.3719
	

	80000/80000 [==============================] - 241s 3ms/step - loss: 1.2913 - acc: 0.6535 - top_k_categorical_accuracy: 0.8886 - val_loss: 7.1783 - val_acc: 0.1801 - val_top_k_categorical_accuracy: 0.3652
	

	80000/80000 [==============================] - 244s 3ms/step - loss: 1.2608 - acc: 0.6585 - top_k_categorical_accuracy: 0.8938 - val_loss: 7.4171 - val_acc: 0.1772 - val_top_k_categorical_accuracy: 0.3654
	

	80000/80000 [==============================] - 245s 3ms/step - loss: 1.2429 - acc: 0.6660 - top_k_categorical_accuracy: 0.8966 - val_loss: 7.3468 - val_acc: 0.1793 - val_top_k_categorical_accuracy: 0.3670
	

	80000/80000 [==============================] - 245s 3ms/step - loss: 1.2206 - acc: 0.6705 - top_k_categorical_accuracy: 0.9003 - val_loss: 7.7206 - val_acc: 0.1839 - val_top_k_categorical_accuracy: 0.3712
	

	80000/80000 [==============================] - 238s 3ms/step - loss: 1.1945 - acc: 0.6777 - top_k_categorical_accuracy: 0.9044 - val_loss: 7.7350 - val_acc: 0.1785 - val_top_k_categorical_accuracy: 0.3648
	

	80000/80000 [==============================] - 241s 3ms/step - loss: 1.1816 - acc: 0.6795 - top_k_categorical_accuracy: 0.9073 - val_loss: 7.8643 - val_acc: 0.1794 - val_top_k_categorical_accuracy: 0.3625
	

	80000/80000 [==============================] - 231s 3ms/step - loss: 1.1680 - acc: 0.6851 - top_k_categorical_accuracy: 0.9097 - val_loss: 8.0242 - val_acc: 0.1809 - val_top_k_categorical_accuracy: 0.3685
	

	80000/80000 [==============================] - 243s 3ms/step - loss: 1.1455 - acc: 0.6898 - top_k_categorical_accuracy: 0.9136 - val_loss: 8.0614 - val_acc: 0.1764 - val_top_k_categorical_accuracy: 0.3644
	

	80000/80000 [==============================] - 258s 3ms/step - loss: 1.1306 - acc: 0.6925 - top_k_categorical_accuracy: 0.9152 - val_loss: 8.1652 - val_acc: 0.1784 - val_top_k_categorical_accuracy: 0.3674
	

	80000/80000 [==============================] - 265s 3ms/step - loss: 1.1163 - acc: 0.6969 - top_k_categorical_accuracy: 0.9178 - val_loss: 8.1587 - val_acc: 0.1764 - val_top_k_categorical_accuracy: 0.3659
	

	80000/80000 [==============================] - 270s 3ms/step - loss: 1.1088 - acc: 0.7013 - top_k_categorical_accuracy: 0.9188 - val_loss: 8.3066 - val_acc: 0.1741 - val_top_k_categorical_accuracy: 0.3579
	

	80000/80000 [==============================] - 260s 3ms/step - loss: 1.0948 - acc: 0.7034 - top_k_categorical_accuracy: 0.9227 - val_loss: 8.3521 - val_acc: 0.1719 - val_top_k_categorical_accuracy: 0.3529
	

	80000/80000 [==============================] - 256s 3ms/step - loss: 1.0910 - acc: 0.7053 - top_k_categorical_accuracy: 0.9235 - val_loss: 8.4043 - val_acc: 0.1704 - val_top_k_categorical_accuracy: 0.3555
	

	80000/80000 [==============================] - 248s 3ms/step - loss: 1.0773 - acc: 0.7081 - top_k_categorical_accuracy: 0.9262 - val_loss: 8.5451 - val_acc: 0.1749 - val_top_k_categorical_accuracy: 0.3617
	

	80000/80000 [==============================] - 248s 3ms/step - loss: 1.0668 - acc: 0.7108 - top_k_categorical_accuracy: 0.9266 - val_loss: 8.5922 - val_acc: 0.1718 - val_top_k_categorical_accuracy: 0.3588
	

	80000/80000 [==============================] - 246s 3ms/step - loss: 1.0596 - acc: 0.7133 - top_k_categorical_accuracy: 0.9293 - val_loss: 8.5660 - val_acc: 0.1652 - val_top_k_categorical_accuracy: 0.3448
	

	80000/80000 [==============================] - 246s 3ms/step - loss: 1.0497 - acc: 0.7182 - top_k_categorical_accuracy: 0.9305 - val_loss: 8.6918 - val_acc: 0.1758 - val_top_k_categorical_accuracy: 0.3589
	

	80000/80000 [==============================] - 248s 3ms/step - loss: 1.0363 - acc: 0.7221 - top_k_categorical_accuracy: 0.9315 - val_loss: 8.8265 - val_acc: 0.1732 - val_top_k_categorical_accuracy: 0.3564
	

	80000/80000 [==============================] - 248s 3ms/step - loss: 1.0340 - acc: 0.7208 - top_k_categorical_accuracy: 0.9330 - val_loss: 8.8335 - val_acc: 0.1741 - val_top_k_categorical_accuracy: 0.3563"""

	#print(data)

	# parse the data
	xs = []
	losses = [] 
	validation_losses = [] 
	lines = data.split('\n')
	x=1
	LOSS_IDX, VALIDATION_LOSS_IDX = 7,16
	for line in lines:
		line = line.strip()
		line_tokens = line.split(' ')
		if len(line_tokens) > 1:
			losses.append(float(line_tokens[LOSS_IDX]))
			validation_losses.append(float(line_tokens[VALIDATION_LOSS_IDX]))
			xs.append(x)
			x += 1

	#print(losses)
	#print(validation_losses)

	xs = xs[:15]
	losses = losses[:15]
	validation_losses = validation_losses[:15]

	"""
	plot_name = 'learning_curve_cnn_{}.png'.format(str(datetime.datetime.now()))
	fig, ax = plt.subplots()
	line1, = ax.plot(xs, losses, label='Training Loss')
	line2, = ax.plot(xs, validation_losses, label='Validation Loss')
	title='CNN Loss Curve'
	ax.legend()
	plt.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
	plt.ylabel('Loss')
	plt.title(title)
	plt.savefig(plot_name)
	"""

	losses = [7.2658, 6.9714, 6.8302, 6.6318, 6.3564, 6.057, 5.7686, 5.4802, 5.1918, 4.9033999999999995, 4.614999999999999, 4.326599999999999, 4.038199999999999, 3.7919, 3.7497999999999987, 3.6284, 3.4893, 3.3693, 3.2688, 3.1787]
	accuracies = [0.0049, 0.0108, 0.0325, 0.0706, 0.1155, 0.1618, 0.2059, 0.25, 0.29410000000000003, 0.33820000000000006, 0.3823000000000001, 0.4264000000000001, 0.47050000000000014, 0.5113, 0.5146000000000002, 0.5353, 0.5563, 0.574, 0.5888, 0.6024]
	assert(len(losses) == len(accuracies))
	batch_nums = list(range(1,len(losses)+1))

	plot_name = 'learning_curve_cnn_{}.png'.format(str(datetime.datetime.now()))
	
	fig, ax1 = plt.subplots()
	
	color = 'tab:red'
	ax1.set_ylabel('loss', color=color)
	ax1.plot(batch_nums, losses, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:blue'
	ax2.set_ylabel('accuracy', color=color)  # we already handled the x-label with ax1
	ax2.plot(batch_nums, accuracies, color=color)
	ax2.tick_params(axis='y', labelcolor=color)

	plt.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom=False,      # ticks along the bottom edge are off
			top=False,         # ticks along the top edge are off
			labelbottom=False) # labels along the bottom edge are off
	
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.savefig(plot_name)

