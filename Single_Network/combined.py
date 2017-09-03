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
        for i in range(shape[2]):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)

def build(input):
    net=slim.conv2d(input,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv1')
    net=slim.conv2d(net,32,[3,3],rate=2,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv2')
    net=slim.conv2d(net,32,[3,3],rate=4,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv3')
    net=slim.conv2d(net,32,[3,3],rate=8,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv4')
    net=slim.conv2d(net,32,[3,3],rate=16,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv5')
    net=slim.conv2d(net,32,[3,3],rate=32,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv6')
    net=slim.conv2d(net,32,[3,3],rate=64,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv7')
    net=slim.conv2d(net,32,[3,3],rate=128,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv8')
    net=slim.conv2d(net,32,[3,3],rate=1,activation_fn=lrelu,normalizer_fn=nm,weights_initializer=identity_initializer(),scope='g_conv9')
    net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_conv_last')
    return net

def prepare_data():
    input_names=[]
    output_names=[]
    train_task=[]
    finetune_input_names=[]
    finetune_output_names=[]
    finetune_task=[]
    val_names=[]
    val_task=[]
    cnt=0
    for task in ['Nonlocal_dehazing','Dark_channel_dehazing','Multiscale_tone_manipulation','Detail_manipulation','L0_smoothing','Pencil_drawing','Rudin_Osher_Fatemi','Relative_total_variation','Photographic_style','TV_L1']:
        for dirname in ['MIT-Adobe_train_480p']:#training images at 480p
            for i in range(1,2501):
                input_names.append("../data/%s/%06d.png"%(dirname,i))
                output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))
                train_task.append(cnt)
        for dirname in ['MIT-Adobe_train_random']:#training images at random resolutions
            for i in range(1,2501):
                finetune_input_names.append("../data/%s/%06d.png"%(dirname,i))
                finetune_output_names.append("../original_results/%s/%s/%06d.png"%(task,dirname,i))
                finetune_task.append(cnt)
        for dirname in ['MIT-Adobe_test_1080p']:#test images at 1080p
            for i in range(1,2501):
                val_names.append("../data/%s/%06d.png"%(dirname,i))
                val_task.append(cnt)
        cnt+=1
    return input_names,output_names,train_task,finetune_input_names,finetune_output_names,finetune_task,val_names,val_task

def one_hot_map(m,n,k):
    tmp=np.zeros((1,m,n,10),dtype=np.float32)
    tmp[:,:,:,k]=(1 if isinstance(k,int) else 1.0/len(k))
    return tmp

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
is_training=False

input_names,output_names,train_task,finetune_input_names,finetune_output_names,finetune_task,val_names,val_task=prepare_data()
input=tf.placeholder(tf.float32,shape=[None,None,None,3+10])
output=tf.placeholder(tf.float32,shape=[None,None,None,3])
network=build(input)
loss=tf.reduce_mean(tf.square(network-output))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

ckpt=tf.train.get_checkpoint_state("result_combined")
if ckpt:
    print('loaded '+ckpt.model_checkpoint_path)
    saver.restore(sess,ckpt.model_checkpoint_path)

if is_training:
    all=np.zeros(30000, dtype=float)
    for epoch in range(1,181):
        if epoch==1 or epoch==151:
            input_images=[None]*len(input_names)
            output_images=[None]*len(input_names)

        if os.path.isdir("result_combined/%04d"%epoch):
            continue
        cnt=0
        for id in np.random.permutation(len(input_names)):
            st=time.time()
            if input_images[id] is None:
                input_images[id]=np.expand_dims(np.float32(cv2.imread(input_names[id] if epoch<=150 else finetune_input_names[id],-1)),axis=0)/255.0
                output_images[id]=np.expand_dims(np.float32(cv2.imread(output_names[id] if epoch<=150 else finetune_output_names[id],-1)),axis=0)/255.0
            if input_images[id].shape[1]*input_images[id].shape[2]>1800000:#GPU memory limitation
                continue

            _,current=sess.run([opt,loss],feed_dict={input:np.concatenate((input_images[id],one_hot_map(input_images[id].shape[1],input_images[id].shape[2],train_task[id] if epoch<=150 else finetune_task[id])),axis=3),output:output_images[id]})
            all[id]=current*255.0*255.0
            cnt+=1
            print("%d %d %.2f %.2f %.2f %s"%(epoch,cnt,current*255.0*255.0,np.mean(all[np.where(all)]),time.time()-st,os.getcwd().split('/')[-2]))
            if cnt>=2500:
                break

        os.makedirs("result_combined/%04d"%epoch)
        target=open("result_combined/%04d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(all[np.where(all)]))
        target.close()

        saver.save(sess,"result_combined/model.ckpt")
        saver.save(sess,"result_combined/%04d/model.ckpt"%epoch)
        for ind in range(10):
            input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
            st=time.time()
            output_image=sess.run(network,feed_dict={input:np.concatenate((input_image,one_hot_map(input_image.shape[1],input_image.shape[2],ind)),axis=3)})
            print("%.3f"%(time.time()-st))
            output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
            cv2.imwrite("result_combined/%04d/%06d.png"%(epoch,ind+1),np.uint8(output_image[0,:,:,:]))

if not os.path.isdir("result_combined/video"):
    os.makedirs("result_combined/video")
input_image=np.expand_dims(np.float32(cv2.imread(val_names[69],-1)),axis=0)/255.0
order=[[7,3],[3,8],[8,[3,7,8]]]
for i in range(3):
    for k in range(100):
        tmp=one_hot_map(input_image.shape[1],input_image.shape[2],order[i][1])*(k/100.0)+one_hot_map(input_image.shape[1],input_image.shape[2],order[i][0])*(1-k/100.0)
        st=time.time()
        output_image=sess.run(network,feed_dict={input:np.concatenate((input_image,tmp),axis=3)})
        print("%.3f"%(time.time()-st))
        output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
        cv2.imwrite("result_combined/video/%06d.png"%(k+i*100),np.uint8(output_image[0,:,:,:]))

exit()
if not os.path.isdir("result_combined/MIT-Adobe_test_1080p"):
    os.makedirs("result_combined/MIT-Adobe_test_1080p")
for ind in range(len(val_names)):
    input_image=np.expand_dims(np.float32(cv2.imread(val_names[ind],-1)),axis=0)/255.0
    st=time.time()
    output_image=sess.run(network,feed_dict={input:np.concatenate((input_image,one_hot_map(input_image.shape[1],input_image.shape[2],val_task[ind])),axis=3)})
    print("%.3f"%(time.time()-st))
    output_image=np.minimum(np.maximum(output_image,0.0),1.0)*255.0
    cv2.imwrite("result_combined/MIT-Adobe_test_1080p/%06d.png"%(ind+1),np.uint8(output_image[0,:,:,:]))

