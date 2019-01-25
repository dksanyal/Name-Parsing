import json
import time
import sys
from pprint import pprint
import math


class HmmDecode3:
    def parse_data(self, filename):
        file = open(filename, 'r')
        data = json.load(file)
        transition_prob = data['transition']
        emission_prob = data['emission']
        unique_tags = data['unique_tags']
        return transition_prob, emission_prob, unique_tags

    def get_tags(self, emission_prob, word):
        tag_list = {}
        for key in emission_prob:
            if key == word:
                tag_list[word] = emission_prob[word]
                break
        return tag_list

    def find_max_tag(self, probability):
        max_tag = list(probability.keys())[0]
        for tag1 in probability:
            if probability[tag1] > probability[max_tag]:
                max_tag = tag1
        return max_tag

    def find_max_tag_backprob(self, probability):
        max_tag = list(probability.keys())[0]
        for tag1 in probability:
            if probability[tag1] > probability[max_tag]:
                max_tag = tag1
        return max_tag

    def smoothing_emission(self, emission, word, unique_tags):
        print(word)
        l = len(list(transition.keys()))
        # prob = - math.log(l)
        prob = 0
        # prob = 1
        emission[word] = {}
        for tag in unique_tags:
            if tag != 'START' and tag != 'STOP':
                emission[word][tag] = []
                emission[word][tag].append(prob)
                emission[word][tag].append(l)
        emission[word]['total_count'] = l
        return emission[word]

    def viterbi(self, transition_prob, emission_prob, unique_tags, sentence):
        result = []
        prob = dict()
        word_list = sentence.split(" ")
        word_count = len(word_list)
        prev_tags = []
        prev_tag_list = []
        backtrack = dict()
        i = 1
        for i in range(1, word_count+1):
            prob[i] = dict()
            backtrack[i] = dict()
            if i == 1:
                # emission smoothing
                if word_list[i-1] not in emission_prob:
                    emission_prob[word_list[i-1]] = self.smoothing_emission(emission_prob, word_list[i-1], unique_tags)
                tag_list = self.get_tags(emission_prob, word_list[i - 1])
                for tag in tag_list[word_list[i-1]]:
                    if tag == 'total_count':
                        continue
                    p = transition_prob['START'][tag][0] + emission_prob[word_list[i-1]][tag][0]
                    # p = transition_prob['START'][tag][0] * emission_prob[word_list[i - 1]][tag][0]
                    # p = transition_prob['START'][tag][0] * tag_list[word_list[i - 1]][tag][0]
                    prob[i][tag] = dict()
                    prob[i][tag]['START'] = p
                    prev_tags.append(tag)
                prev_tag_list = prev_tags
            else:
                prev_tags = prev_tag_list
                prev_tag_list = []
                # emission smoothing
                if word_list[i - 1] not in emission_prob:
                    emission_prob[word_list[i - 1]] = self.smoothing_emission(emission_prob, word_list[i - 1], unique_tags)
                tag_list = self.get_tags(emission_prob, word_list[i - 1])
                for c_tag in tag_list[word_list[i-1]]:
                    if c_tag == 'total_count':
                        continue
                    prob[i][c_tag] = dict()
                    for p_tag in prev_tags:
                        max_tag = self.find_max_tag(prob[i-1][p_tag])
                        p = prob[i-1][p_tag][max_tag] + transition_prob[p_tag][c_tag][0] + emission_prob[word_list[i-1]][c_tag][0]
                        # p = prob[i - 1][p_tag][max_tag] * transition_prob[p_tag][c_tag][0] * emission_prob[word_list[i - 1]][c_tag][0]
                        # p = prob[i - 1][p_tag][max_tag] * transition_prob[p_tag][c_tag][0] * tag_list[word_list[i - 1]][c_tag][0]
                        backtrack[i-1][p_tag] = max_tag
                        prob[i][c_tag][p_tag] = p
                    prev_tag_list.append(c_tag)
        # for last word
        prob[i]['STOP'] = dict()

        for p_tag in prev_tag_list:
            max_tag = self.find_max_tag(prob[i][p_tag])
            p = prob[i][p_tag][max_tag] + transition_prob[p_tag]['STOP'][0]
            # p = prob[i][p_tag][max_tag] * transition_prob[p_tag]['STOP'][0]
            backtrack[i][p_tag] = max_tag
            prob[i]['STOP'][p_tag] = p
        max_tag = self.find_max_tag(prob[i]['STOP'])

        # backtracking
        backtrack[i+1] = dict()
        backtrack[i+1]['STOP'] = max_tag
        l = len(list(backtrack.keys()))
        tag = max_tag
        for i in range(l, 1, -1):
            if i == l:
                result.append([word_list[i - 2], max_tag])
                tag = backtrack[i]['STOP']
            else:
                tag = backtrack[i][tag]
                result.append([word_list[i - 2], tag])
        return result

    def tag_sentences(self, transition_prob, emission_prob, unique_tags, filename):
        file = open(filename, 'r')
        lines = file.read()
        sentences = ""
        for line in lines.split("\n"):
            line = line.strip(" ")
            res = self.viterbi(transition_prob, emission_prob, unique_tags, line)
            res_len = len(res)
            sentence = ""
            for i in range(res_len - 1, -1, -1):
                if i != res_len - 1:
                    sentence += " "
                sentence += res[i][1]
            sentences += sentence + "\n"
        sentences = sentences.strip("\n")
        return sentences

    def write_result(self, res, filename):
        file = open(filename, 'w')
        file.write(res)
        return


t1 = time.time()
if __name__ == "__main__":
    model = HmmDecode3()
    filename1 = "hmmmodel.json"
    filename2 = sys.argv[1]
    filename3 = "hmmoutput.txt"

    transition, emission, unique_tags = model.parse_data(filename1)
    result = model.tag_sentences(transition, emission, unique_tags, filename2)
    model.write_result(result, filename3)
t2 = time.time()
print("Time: " + str(t2 - t1))
