people = ['man', 'woman', 'boy', 'girl', 'people', 'men']
colors = ['orange', 'green', 'red', 'white', 'black', 'pink', 'blue', 'purple', 'tan', 'grey', 'gray', 'yellow', 'gold', 'golden', 'dark', 'brown', 'silver']

if __name__ == "__main__":
    out_color = open('data/masking/data/multi30k.color.position', 'w', encoding='utf-8')
    out_people = open('data/masking/data/multi30k.people.position', 'w', encoding='utf-8')
    with open('data/multi30k/multi30k.en', 'r', encoding='utf-8') as f:
        for sentence in f:
            sentence = sentence.strip().split()
            flag_color = False
            flag_people = False
            for idx, i in enumerate(sentence):
                if i in colors:
                    out_color.write(str(idx)+' ')
                    flag_color = True
                elif i in people:
                    out_people.write(str(idx)+' ')
                    flag_people = True

            if flag_color == False:
                out_color.write(str(-1))
            if flag_people == False:
                out_people.write(str(-1))

            out_color.write('\n')
            out_people.write('\n')

    out_people.close()
    out_color.close()