import csv
a=[]
csvfile=open('CAND_ELIM.CSV','r')
reader=csv.reader(csvfile)
for row in reader:
    a.append(row)
    print(row)

num_attributes=len(a[0])-1
print("Initial hypothesis is")
s=['0']*num_attributes
g=['?']*num_attributes
print("The most specific hypothesis : ",s)
print("The most general hypothesis : ",g)

for j in range(0,num_attributes):
    s[j]=a[0][j]
print("The candidate algorithm")
temp=[]
for i in range(0,len(a)):
   if(a[i][num_attributes]=='yes'):
       for j in range(0,num_attributes):
           if(a[i][j]!=s[j]):
               s[j]='?'
       for j in range(0,num_attributes):
           for k in range(1,len(temp)):
               if temp[k][j]!='?' and temp[k][j]!=s[j]:
                   del temp[k]
       print("For instance{0} the hypothesis is s{0}".format(i+1),s)
   if(len(temp)==0):
           print("For instance{0} the hypothesis is g{0}".format(i+1),g)
   else:
          print("For instance{0} the hypothesis is g{0}".format(i+1),temp)
   if(a[i][num_attributes]=='no'):
     for j in range(0,num_attributes):
      if(s[j]!=a[i][j] and s[j]!='?'):
           g[j]=s[j]
           temp.append(g)
           g=["?"]*num_attributes
print("For instance {0} the hypothesis is s{0}".format(i+1),s)
print("For instance {0} the hypothesis is g{0}".format(i+1),temp)