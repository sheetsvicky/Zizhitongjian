# Zizhitongjian

Outstanding personalities in a semantic analysis of Zizhitongjian

1 Segment Zizitongjian (https://en.wikipedia.org/wiki/Zizhi_Tongjian), a great work in Chinese historiography led by Sima Guang. It records Chinese history from 403 BC to 959 AD, covering 16 dynasties and about 3 million Chinese characters.

1.1 Initiate segmentation with jieba (https://github.com/fxsjy/jieba).
1.2 Remove symbols and stop words. Stop words are manually chosen from top 1000 high-frequency words after segmentation.
1.3 Unify words that represent the same people of followings ('吴起','韩信','霍去病','邓禹','关羽','诸葛亮','桓温','王猛','秦叔宝'). Also do this for negative controls ('赵高','贾充','李林甫'). 

Example: 
温与秦丞相雄等战于白鹿原，温兵不利 ==>
'温', '与', '秦丞相', '雄', '等', '战', '于', '白鹿原', '，', '温', '兵', '不利' ==>
'桓温', '秦丞相', '雄', '战', '白鹿原', '桓温', '兵', '不利'

2 Word2vec analysis (https://en.wikipedia.org/wiki/Word2vec), neural networks that reconstruct words to vector in high-dimensional spaces.

2.1 Implement Word2vec in gensim (https://radimrehurek.com/gensim/index.html)
2.2 Visualize Posthumous names from (https://zh.wikipedia.org/wiki/%E8%B0%A5%E5%8F%B7) by t-SNE and choose indices for evaluting personality ('义','智','忠','德','勇','仁').
![alt tag](https://github.com/sheetsvicky/Zizhitongjian/blob/master/ft.png)

3 Compare personalities

3.1 Visualize personalities by t-SNE
![alt tag](https://github.com/sheetsvicky/Zizhitongjian/blob/master/pp.png)

3.2 Compare them by similarity score of indices
![alt tag](https://github.com/sheetsvicky/Zizhitongjian/blob/master/pp_radar.png)

3.3 Interesting results
![alt tag](https://github.com/sheetsvicky/Zizhitongjian/blob/master/similar1.png)
![alt tag](https://github.com/sheetsvicky/Zizhitongjian/blob/master/similar2.png)
