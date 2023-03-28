一、环境：
首先，创建一个Conda环境，并为运行实验安装一些必要的软件包。
TensorFlow 2.0
pandas库
Numpy库
Openbabel软件
mdtraj库

二、数据准备
验证集，训练集，测试集
(1)PDBbind v2020所有数据真实pKa来自于文件"INDEX_general_PL_data.2020"
(2)所有文件配体的.mol2文件经过openbabel转换成 .pdb，保留转换没有报错的文件
(3)截取pka值分布在2-12范围内的数据，考虑在可承受范围内具有已知解离常数或抑制常数的复合物(pKi和 pKd值分布在 2-12 范围内)
(4)PDBbind2020 中的复合体排除CASF-2013，CASF-2016数据集的数据
   排除CASF-2013（161个）重复文件后剩余14860
   排除CSAF-2016（254个）重复文件后剩余14696
   确定数据个数：	训练集：12000个
	           	测试集：2827个
		验证集CASF-2013：161个 
		验证集CASF-2016：254个
  
三、文件处理
(1)调用"生成特征.py"文件，生成输入特征文件:	"Onion1_Feature_2020_all_train.csv"
				 	"Onion1_Feature_2020_all_valid.csv"
					"Onion1_Feature_2013.csv"
					"Onion1_Feature_2016.csv"

(2)调用"连接数据和pka.py"文件，连接生成的特征和蛋白质配体复合物的pka值，生成文件： 	"Onion1_Feature_2020_all_pka_train.csv"
									"Onion1_Feature_2020_all_pka_valid.csv"
	
(3)调用"训练网络.py"，训练得到模型："bestmodel.h5"，"logfile.log"

(4)调用"预测.py"，得到测试集的预测结果："",""



