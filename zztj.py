# -*- coding: utf-8 -*-
# 2016.10.9
pip install --upgrade gensim
pip install jieba
# run python in windows command
# python -i or ipython (but ipython does not go well in emacs)
import jieba
import gensim

# 2016.10.12, for installing gensim on windows
# install NumPy
# install SciPy from "http://www.lfd.uci.edu/~gohlke/pythonlibs"
# install gensim

# read text
# convert encoding in shell
# iconv -c -f GB2312 -t UTF-8 zztj.txt >> zztj_mac.txt
# on windows, one can directly use Notepad save as and choose encoding.

with open('zztj_utf8.txt','rb') as f:
    raw=[str.decode('utf-8').rstrip('\r\n') for str in f.readlines()] # length: 31698

anno=raw[129:132]
zztj=[str for str in raw if str not in anno] # length: 29936, notice many items begin with "\u3000"

# try jieba to segment words
import jieba
print(", ".join(jieba.cut(zztj[6],HMM=False))) # HMM=false represses new words
seg=jieba.cut(zztj[1],HMM=False)
print(" ".join(seg)) # run this line again, it will return blank! once-only generator stream?

# 2016.10.15, define my dictionary for jieba: my_dict.txt (must be utf-8)
jieba.load_userdict('my.dt')
sym=["，","。","《","》","：","“","”","、","；","\u3000","\ufeff",":",'？','！',' ','\ue345','＊','\ue767','\ue4bf','‘','’','\ue3ab']
zztj_sep=list(jieba.lcut(sen,HMM=False) for sen in zztj[1:127]) # return list, 1:126 is the first document of Zhou
doc=[ [word for word in sen if word not in sym] for sen in zztj_sep] # nested list

# word2vec
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)xo
model=word2vec.Word2Vec(doc,min_count=1)
# 'doc' must be a list of sentences, otherwise Chinese words will be segmented into single character.
model.most_similar(positive=['天子'],negative=[])
model.most_similar(positive=['天子','卿大夫'],negative=['三公'])

# 2016.10.17, general model
def build_model(x):
    zztj_sep=list(jieba.lcut(sen,HMM=False) for sen in x) 
    doc=[ [word for word in sen if word not in sym] for sen in zztj_sep] 
    model=word2vec.Word2Vec(doc,min_count=1)
    return model

model=build_model(zztj) # input all
model.save('plain_model')

# 2016.10.23, test case of 关羽, make name consistent.
# raw text
zztj_sep=list(jieba.lcut(sen,HMM=False) for sen in zztj) 
doc=[ [word for word in sen if word not in sym] for sen in zztj_sep]
doc2=doc.copy() # Create a copy of the list to update names in doc2. Do not use "=".

gy_ind=list(zztj.index(x) for x in ['汉纪 汉纪五十二','魏纪 魏纪一']) # [5286, 5783], begin and end (not included) of Guan Yu
gy_sep=doc2[gy_ind[0]:gy_ind[1]]
gy_sub=[ [word if word != '羽' else '关羽' for word in sen ] for sen in gy_sep] # substitute '羽' to '关羽'
tmp=[sen.count('项羽') for sen in gy_sub] # some '羽' is '项羽'
[x for x in range(len(tmp)) if tmp[x]>0 ] # [170, 233]
gy_sub[170][gy_sub[170].index('关羽')]='项羽'
# use gy_sub in raw document
doc2[gy_ind[0]:gy_ind[1]]=gy_sub

# 秦琼
qq_ind=list(zztj.index(x) for x in ['隋纪 隋纪六','唐纪 唐纪十三']) # [17507, 19107]
qq_sep=doc2[qq_ind[0]:qq_ind[1]] 
qq_sub=[ [word if word != '叔宝' else '秦叔宝' for word in sen ] for sen in qq_sep]
doc2[qq_ind[0]:qq_ind[1]]=qq_sub


# 2016.10.24, 孙膑
# all names are unique

# 吴起
wq_ind=list(zztj.index(x) for x in ['周纪 周纪一','周纪 周纪二']) #
wq_sep=doc2[wq_ind[0]:wq_ind[1]] 
wq_sub=[ [word if word != '起' else '吴起' for word in sen ] for sen in wq_sep]
doc2[wq_ind[0]:wq_ind[1]]=wq_sub

# 韩信
hx_ind=list(zztj.index(x) for x in ['汉纪 汉纪一','汉纪 汉纪五']) # [815, 1071]
# doc[833] can not separate '听信'
hx_sep=doc2[hx_ind[0]:hx_ind[1]] 

def list_contain(ls,str):
    if str in ls: return True

def all_index(ls,str):
    return([x for x in range(len(ls)) if ls[x]==str])

def list_list_contain(ls,str):
    return([x for x in range(len(ls)) if list_contain(ls[x],str)])

ind=[x for x in range(len(hx_sep)) if list_contain(hx_sep[x],'信') ] # all index in "hx_sep" contains '信', check and record negative indices
not_ind=[(0,49),(4,75),(5,171),(8,83),(10,48),(10,125),(11,82),(12,90),(16,337),(16,468),(19,482),(28,440),(28,451),(28,460),(28,461),(29,16),(29,40),(30,53),(30,64),(30,93),(34,37)] # One should be careful about index because update dictionary and re-segmentation may change those indices
list(hx_sep[ind[x]][y] for x,y in not_ind) # check whether all are '信'
for x,y in not_ind: hx_sep[ind[x]][y]='xin' 

hx_sub=[ [word if word != '信' else '韩信' for word in sen ] for sen in hx_sep]
hx_sub=[ [word if word != 'xin' else '信' for word in sen ] for sen in hx_sub] # set 'xin' back to '信'

doc2[hx_ind[0]:hx_ind[1]]=hx_sub

# 2016.10.25, 霍去病
hqb_ind=list(zztj.index(x) for x in ['汉纪 汉纪十一','汉纪 汉纪十三']) # 
hqb_sep=doc2[hqb_ind[0]:hqb_ind[1]] 
hqb_sub=[ [word if word != '去病' else '霍去病' for word in sen ] for sen in hqb_sep]
doc2[hqb_ind[0]:hqb_ind[1]]=hqb_sub

# 诸葛亮
zgl_ind=list(zztj.index(x) for x in ['汉纪 汉纪五十七','魏纪 魏纪五']) # 
zgl_sep=doc2[zgl_ind[0]:zgl_ind[1]] 
zgl_sub=[ [word if word != '亮' else '诸葛亮' for word in sen ] for sen in zgl_sep]
zgl_sub=[ [word if word != '诸葛孔明' else '诸葛亮' for word in sen ] for sen in zgl_sub] # second substitute should based on previous version
zgl_sub=[ [word if word != '孔明' else '诸葛亮' for word in sen ] for sen in zgl_sub]
doc2[zgl_ind[0]:zgl_ind[1]]=zgl_sub

# 桓温, omit 桓宣武
hw_ind=list(zztj.index(x) for x in ['晋纪 晋纪十九','晋纪 晋纪二十六']) # 
hw_sep=doc2[hw_ind[0]:hw_ind[1]] 
hw_sub=[ [word if word != '温' else '桓温' for word in sen ] for sen in hw_sep]
doc2[hw_ind[0]:hw_ind[1]]=hw_sub

# 王猛
wm_ind=list(zztj.index(x) for x in ['晋纪 晋纪二十一','晋纪 晋纪二十六']) # 
wm_sep=doc2[wm_ind[0]:wm_ind[1]] # notice indices of different people will overlap, always take 'doc' from last updated version
wm_sub=[ [word if word != '猛' else '王猛' for word in sen ] for sen in wm_sep]
wm_sub=[ [word if word != '王景略' else '王猛' for word in sen ] for sen in wm_sub]
doc2[wm_ind[0]:wm_ind[1]]=wm_sub

# 郭子仪
gzy_ind=list(zztj.index(x) for x in ['唐纪 唐纪三十二','唐纪 唐纪四十四']) # 
gzy_sep=doc2[gzy_ind[0]:gzy_ind[1]]
gzy_sub=[ [word if word != '子仪' else '郭子仪' for word in sen ] for sen in gzy_sep]
doc2[gzy_ind[0]:gzy_ind[1]]=gzy_sub

# 2016.10.26, 刘备
lb_ind=list(zztj.index(x) for x in ['汉纪 汉纪五十二','魏纪 魏纪三']) # 
lb_sep=doc2[lb_ind[0]:lb_ind[1]]
# segment 益备 备警 备礼 备御 为备 备物, can not segment 设备 
not_ind=[(228,53),(273,29),(77,28),(500,276),(500,326),(548,281),(298,133),(169,168),(298,178),(372,61),(372,138),(389,73)]
list(lb_sep[x][y] for x,y in not_ind) # check whether all are '备'
for x,y in not_ind: lb_sep[x][y]='bei'

lb_sub=[ [word if word != '备' else '刘备' for word in sen ] for sen in lb_sep]
lb_sub=[ [word if word != '玄德' else '刘备' for word in sen ] for sen in lb_sub]
lb_sub=[ [word if word != '刘玄德' else '刘备' for word in sen ] for sen in lb_sub]
for x,y in not_ind: lb_sub[x][y]='备'

doc2[lb_ind[0]:lb_ind[1]]=lb_sub

# 苻坚
fj_ind=list(zztj.index(x) for x in ['晋纪 晋纪二十一','晋纪 晋纪二十九']) # 
fj_sep=doc2[fj_ind[0]:fj_ind[1]]
# 贾坚 壁坚 未坚 南坚
not_ind=[(224,13),(224,23),(224,24),(224,34),(224,59),(224,117),(224,119),(224,156),(224,173),(224,184),(224,197),(224,242),(224,256),(437,287),(389,28),(214,27)]
list(fj_sep[x][y] for x,y in not_ind) # check whether all are ''
for x,y in not_ind: fj_sep[x][y]='jian'

fj_sub=[ [word if word != '坚' else '苻坚' for word in sen ] for sen in fj_sep]
for x,y in not_ind: fj_sub[x][y]='坚'

doc2[fj_ind[0]:fj_ind[1]]=fj_sub

# 2016.10.27, 贾充
jc_ind=list(zztj.index(x) for x in ['魏纪 魏纪八','晋纪 晋纪四']) # 
jc_sep=doc2[jc_ind[0]:jc_ind[1]]
jc_sep[544][31]='chong'
jc_sub=[ [word if word != '充' else '贾充' for word in sen ] for sen in jc_sep]
jc_sub[544][31]='充'
doc2[jc_ind[0]:jc_ind[1]]=jc_sub

# 邓禹
dy_ind=list(zztj.index(x) for x in ['汉纪 汉纪三十一','汉纪 汉纪三十七']) # 
dy_sep=doc2[dy_ind[0]:dy_ind[1]]
dy_sep[198][71]='yu'
dy_sub=[ [word if word != '禹' else '邓禹' for word in sen ] for sen in dy_sep]
dy_sub[198][71]='禹'
doc2[dy_ind[0]:dy_ind[1]]=dy_sub

# 赵高
zg_ind=list(zztj.index(x) for x in ['秦纪 秦纪二','汉纪 汉纪一']) # 
zg_sep=doc2[zg_ind[0]:zg_ind[1]]
zg_sub=[ [word if word != '高' else '赵高' for word in sen ] for sen in zg_sep]
doc2[zg_ind[0]:zg_ind[1]]=zg_sub

# 李林甫
llf_ind=list(zztj.index(x) for x in ['唐纪 唐纪二十九','唐纪 唐纪四十一']) # 
llf_sep=doc2[llf_ind[0]:llf_ind[1]]
llf_sub=[ [word if word != '林甫' else '李林甫' for word in sen ] for sen in llf_sep]
doc2[llf_ind[0]:llf_ind[1]]=llf_sub


# 2016.10.28
# count words in segmented documents and record stop words
flat_doc=[x for sen in doc2 for x in sen]
from collections import Counter
doc_count=dict(Counter(flat_doc))
len(list(doc_count.keys())) # number of unique words
import operator
sort_count = sorted(doc_count.items(), key=operator.itemgetter(1),reverse=True)
# output high-frequent words and manually edit 
with open('stop_word.txt','w') as f:
    f.writelines(x+'\n' for x,y in sort_count[0:1000])

# remove stop words
with open('stop_word_utf8.txt','rb') as f:
    stop=[str.decode('utf-8').rstrip('\r\n') for str in f.readlines()]

doc3=[ [word for word in sen if word not in stop] for sen in doc2]
          
# make model
model=word2vec.Word2Vec(doc3,min_count=15,window=5)
model.save('plain_model')
model = word2vec.Word2Vec.load('plain_model')

ft_can=['德','忠','文','武','明','智','达','贤','勤','仁','义','正','勇','信','威','良']
ft=['义','智','忠','德','勇','仁']
pl_can=['吴起','韩信','霍去病','邓禹','关羽','诸葛亮','桓温','王猛','秦叔宝']
ng_can=['赵高','贾充','李林甫']
pp_can=pl_can+ng_can

# t-SNE
import numpy as np
from sklearn.manifold import TSNE
ft_vec=model[ft_can]
pp_vec=model[pp_can]
pp_tsne = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
pp_res=pp_tsne.fit_transform(pp_vec)

# plot
import matplotlib.pyplot as plt # cannot run matplotlib in windows emacs! Why?
#from matplotlib.backends.backend_pdf import PdfPages
#from __future__ import unicode_literals
from matplotlib.font_manager import FontProperties
ft_col=['red' if list_contain(ft,x) else 'black' for x in ft_can]
pp_col=['red' if list_contain(ng_can,x) else 'black' for x in pp_can]
f=plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
ax.axis([min(pp_res[:,0])-0.00001, max(pp_res[:,0])+0.00001, min(pp_res[:,1])-0.00001, max(pp_res[:,1])+0.00001])
#ax.scatter(pp_res[:, 0], pp_res[:, 1])
for label, x, y,col in zip(pp_can, pp_res[:, 0], pp_res[:, 1],pp_col):
    ax.text(x,y,label, fontsize=15,fontproperties=('SimHei'),horizontalalignment='center',verticalalignment='center',color=col)

# save as pdf
dev = PdfPages('pp.pdf')
f.savefig(dev, format='pdf')
dev.close()
# save as png
f.savefig('plot/pp.png', format='png',dpi=200)

# 2016.10.29, similarity
score=[ [model.similarity(x,y) for y in ft] for x in pp_can]
score_lb=[ [(x,y) for y in ft] for x in pp_can ]
model.most_similar(positive=['诸葛亮','苻坚'],negative=['刘备'])
model.most_similar(positive=['诸葛亮'])

# 2016.10.30, radar plot
exec(open("radar.py").read()) # load function from .py
import math
N = 6
colors = ['b', 'r', 'g', 'm', 'y','k']
data=[(x,y) for x,y in zip(pp_can,score)]
theta = radar_factory(N, frame='polygon')
fig = plt.figure(figsize=(9, 9))
fig.subplots_adjust(wspace=0.05, hspace=0.5, top=0.8, bottom=0.1)
for n, (title, case_data) in enumerate(data):
    ax = fig.add_subplot(4, 3, n + 1, projection='radar') # 
    plt.rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='medium', position=(0, 0.9),horizontalalignment='center', verticalalignment='center',fontproperties=('SimHei'),color='b')
    ax.set_ylim([0,0.6])
    ax.plot(theta, case_data) 
    ax.set_varlabels(ft)

plt.tight_layout()
plt.savefig('plot/pp_radar.png', format='png',dpi=200)
