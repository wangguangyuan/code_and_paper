# -*- coding: utf-8 -*-
'''
平台名称  :礼德财富
平台网址  :http://www.leadercf.com/
函数的作者: 王光远
'''
import myCrawler
from bs4 import BeautifulSoup

'''网站公告信息抓取'''
def get_htlm(htlm):
    gonggao_dict={'网站公告中最后一次公告的网址':None,'网站公告中最后一次公告的标题':None,'网站公告中最后一次公告的时间':None}
    
    '''定义一个BeautifulSoup'''
    soup=BeautifulSoup(htlm,"lxml")
    
#    mysoup_htlm=soup.prettify()
#    print(mysoup_htlm)
    a_list=soup.find_all('a',{'class':"fl"})
    gonggao_dict['网站公告中最后一次公告的网址']=a_list[0].attrs['href']
    span_list=soup.find_all('span',{'class':"announce-title"})
    gonggao_dict['网站公告中最后一次公告的标题']=span_list[0].string
    span_list=soup.find_all('span',{'class':"date-time fr mr30"})
    gonggao_dict['网站公告中最后一次公告的时间']=span_list[0].string
    return gonggao_dict



'''抓取网页数据'''
def get_data(htlm):
    data=0
    data_list=[]
    data_dict={}
    '''定义一个BeautifulSoup'''
    soup=BeautifulSoup(htlm,"lxml")
    
    '''通过改变传入的参数 'div',{'class':"textValue"} 来该变爬去的内容 '''
    tag_list=soup.find_all('div',{'class':"textValue"})  
    
    ''' 通过改变传入的参数 'div',{'class':"textValue"} 来该变爬去的内容'''
    world_list=soup.find_all(attrs={'class':"text"})   
    '''看找到结果是否正确，通过打开#，看结果'''
    
#    for each in tag_list:
#        print(each)
#    for each in world_list:
#        print(each)
#    
    for each in range(len(tag_list)):
        for i in tag_list[each].strings:
            data_list.append(i)
#            '''查看data_list里的数据，去掉#''''
#        print(data_list)
        data=deal_data(data_list)
        data_dict[world_list[each].string]=data
    return data_dict


'''数据转换'''   
def deal_data(data_list):
    sum_data=0
    mydata_list=[]
    myyuan_list=[]
    for each in range(len(data_list)):
        if(each%2==0):
            mydata_list.append(data_list[each])
        else:
              myyuan_list.append(data_list[each])
    '''测试数据是否分离成功，可以去掉#,看结果'''
#    print(mydata_list)
#    print(myyuan_list)
    for each in range(len(myyuan_list)): 
        if(myyuan_list[each]=='亿'):
             sum_data=sum_data+int(mydata_list[each])*1e9
        elif(myyuan_list[each]=='万'):
            sum_data=sum_data+int(mydata_list[each])*1e4
        elif (myyuan_list[each]=='千'):
            sum_data=sum_data+int(mydata_list[each])*1e3
        elif (myyuan_list[each]=='元'):
            sum_data=sum_data+int(mydata_list[each])
    return sum_data
            
   
    
    
if __name__=='__main__':
    url_list=['http://www.leadercf.com/aboutus/outcomes','http://www.leadercf.com/article/page/3773/16/1']
    htlm=myCrawler.open_url(url_list[1])
    gonggao_dict=get_htlm(htlm)
    print(gonggao_dict)
    htlm=myCrawler.open_url(url_list[0])
    mydata_dict=get_data(htlm)
    print(mydata_dict)

