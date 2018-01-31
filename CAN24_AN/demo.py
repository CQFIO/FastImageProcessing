from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def lrelu(x):
    return tf.maximum(x*0.2,x)

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x) # the parameter "is_training" in slim.batch_norm does not seem to help so I do not use it

def build(input):
    net=slim.conv2d(input,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,24,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,24,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,24,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,24,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,24,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,24,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
#    net=slim.conv2d(net,24,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')
    net=slim.conv2d(net,24,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return net

def prepare_data(task):
    input_names=[]
    output_names=[]
    finetune_input_names=[]
    finetune_output_names=[]
    val_names=[]
    for dirname in ['MIT-Adobe_train_480p']:
        for i in range(1,2501):
            input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at 480p
            output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at 480p
    for dirname in ['MIT-Adobe_train_random']:
        for i in range(1,2501):
            finetune_input_names.append("../data/%s/%06d.png"%(dirname,i))#training input image at random resolution
            finetune_output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))#training output image at random resolution
    for dirname in ['MIT-Adobe_test_1080p']:
        for i in range(1,2501):
            val_names.append("../data/%s/%06d.png"%(dirname,i))#test input image at 1080p
    return input_names,output_names,val_names,finetune_input_names,finetune_output_names

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

task="L0_smoothing"
is_training=False
sess=tf.Session()

input_names,output_names,val_names,finetune_input_names,finetune_output_names=prepare_data(task)
input=tf.placeholder(tf.float32,shape=[None,None,None,3])
output=tf.placeholder(tf.float32,shape=[None,None,None,3])
network=build(input)
loss=tf.reduce_mean(tf.square(network-output))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss,var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state(task)
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if is_training:
    all=np.zeros(3000, dtype=float)
    for epoch in range(1,181):
        if epoch==1 or epoch==151:
            input_images=[None]*len(input_names)
            output_images=[None]*len(input_names)

        if os.path.isdir("%s/%04d"%(task,epoch)):
            continue
        cnt=0
        for id in np.random.permutation(len(input_names)):
            st=time.time()
            if input_images[id] is None:
                input_images[id]=np.expand_dims(np.float32(cv2.imread(input_names[id] if epoch<=150 else finetune_input_names[id],-1)),axis=0)/255.0
                output_images[id]=np.expand_dims(np.float32(cv2.imread(output_names[id] if epoch<=150 else finetune_output_names[id],-1)),axis=0)/255.0
            if input_images[id].shape[1]*input_images[id].shape[2]>2200000:#due to GPU memory limitation
                continue
            _,current=sess.run([opt,loss],feed_dict={input:input_images[id],output:output_images[id]})
            all[id]=current*255.0*255.0
            cnt+=1
            print("%d %d %.2f %.2f %.2f %s"%(epoch,cnt,current*255.0*255.0,np.mean(all[np.where(all)]),time.time()-st,os.getcwd().split('/')[-2]))

        os.makedirs("%s/%04d"%(task,epoch))
        target=open("%s/%04d/score.txt"%(task,epoch),'w')
        target.write("%f"%np.mean(all[np.where(all)]))
        target.close()

        saver.save(sess,"%s/model.ckpt"%task)
        saver.save(sess,"%s/%04d/model.ckpt"%(task,epoch))
        for ind in range(10):
            input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
            st=time.time()
            output_image=sess.run(network,feed_dict={input:input_image})
            print("%.3f"%(time.time()-st))
            output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
            cv2.imwrite("%s/%04d/%06d.jpg"%(task,epoch,ind+1),np.uint8(output_image[0,:,:,:]))

if not os.path.isdir("%s/MIT-Adobe_test_1080p_result"%task):
    os.makedirs("%s/MIT-Adobe_test_1080p_result"%task)
for ind in range(len(val_names)):
    if not os.path.isfile(val_names[ind]):
        continue
    input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
    st=time.time()
    output_image=sess.run(network,feed_dict={input:input_image})
    print("%.3f"%(time.time()-st))
    output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
    cv2.imwrite("%s/MIT-Adobe_test_1080p_result/%06d.png"%(task,ind+1),np.uint8(output_image[0,:,:,:]))

