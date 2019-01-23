import sys
class ResultCompare:
    def compare(self, filename1, filename2):
        file1 = open(filename1, 'r')      
        file2 = open(filename2, 'r')     
        content1=file1.readlines()
        content2=file2.readlines()
        content1=[x.strip() for x in content1]
        content2=[x.strip() for x in content2]
        print(len(content1))
        print(len(content2))
        correct_count = 0
        incorrect_count = 0
        for k in range(0,len(content2)):
            i = 0
            
            if content1[k] == content2[k]:
                    correct_count += 1
                    
            else:
                    incorrect_count += 1
        print(incorrect_count+correct_count)
        print(float(correct_count)/(correct_count+incorrect_count))


if __name__ == '__main__':
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    model = ResultCompare()
    model.compare(file1, file2)
