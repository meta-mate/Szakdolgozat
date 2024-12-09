using System.Collections;
using System.Collections.Generic;

public interface NodeValue{
    T deriveImplication<T>(List<T> nodeValues) where T : NodeValue;
    string ToString();
}

public class PatternReader<T> where T : NodeValue
{

    public class Node{
        T value, firstValue;
        List<T> values;
        List<int> name, bigger, incremented, lesserFromBigger;
        bool hasBigger = false;

        const int leftBracket = int.MinValue,
        rightBracket = int.MaxValue;

        public Node(List<int> name, T value){
            this.name = name;
            bigger = calculateBigger();
            incremented = calculateIncremented();

            this.value = value;
            firstValue = value;
            values = new List<T>();
            values.Add(firstValue);
        }

        public Node(List<int> name){
            this.name = name;
            bigger = calculateBigger();
            incremented = calculateIncremented();
            values = new List<T>();
        }

        public List<int> getName(){
            return name;
        }

        public List<int> getBigger(){
            return bigger;
        }

        public bool getHasBigger(){
            return hasBigger;
        }

        public List<int> getLesserFromBigger(){
            return lesserFromBigger;
        }

        public List<int> getIncremented(){
            return incremented;
        }

        public List<int> copyName(){
            List<int> r = new List<int>();
            
            for(int i = 0; i < name.Count; i++){
                r.Add(name[i]);
            }

            return r;
        }

        public T getValue(){
            return value;
        }

        public T getFirstValue(){
            return firstValue;
        }

        public List<T> getValues(){
            return values;
        }

        public void setFirstValue(T firstValue){
            this.firstValue = firstValue;
        }

        public void setValue(T value){
            this.value = value;
            if(firstValue == null){
                firstValue = value;
            }
            values.Add(value);
        }

        string intToString(int x){
            string str = "";
            if(x == leftBracket){
                str += "(";
            }else if(x == rightBracket){
                str += ")";
            }else{
                str += x + ".";
            }
            return str;
        }

        public string valuesToString(){
            string str = "[";

            for(int i = 0; i < values.Count(); i++){
                str += values[i].ToString();
                if(i < values.Count - 1) str += " ";
            }

            str += "]";

            return str;
        }

        public string nameToString(){
            if(name == null) return null;
            
            string str = "";
            for(int i = name.Count - 1; i >= 0; i--){
                str += intToString(name[i]);
            }
            if(str[str.Length - 1] == '.') str = str.Remove(str.Length - 1);

            //if(hasBigger) str += "hasBigger";

            return str;
        }

        public string nameToString(List<int> name){
            if(name == null) return null;
            
            string str = "";
            for(int i = name.Count - 1; i >= 0; i--){
                str += intToString(name[i]);
            }
            if(str[str.Length - 1] == '.') str = str.Remove(str.Length - 1);

            return str;
        }

        public bool nameEquals(List<int> name){
            if(this.name.Count != name.Count) return false;

            for(int i = 0; i < name.Count; i++){
                if(this.name[i] != name[i]){
                    return false;
                }
            }

            return true;
        }

        List<int> calculateIncremented(){
            List<int> r = copyName();

            r.Insert(1, 1);

            return r;
        }

        List<int> calculateBigger(){
            List<int> r = new List<int>(){1};
            lesserFromBigger = new List<int>(){1};

            hasBigger = false;
            int lastSameIndex = 0;
            for(int i = 2; i < name.Count; i++){
                if(name[i] == name[i - 1]){
                    lastSameIndex = i;
                    lesserFromBigger.Add(name[i]);
                    hasBigger = true;
                }else break;
            }

            for(int i = lastSameIndex + 1; i < name.Count; i++){
                r.Add(name[i]);
                lesserFromBigger.Add(name[i]);
            }

            if(lastSameIndex != 0)r.Insert(1, name[lastSameIndex] + 1);

            bigger = r;
            return r;
        }

        public bool biggerThan(List<int> name){
            int max = 0;
            for(int i = 0; i < name.Count; i++){
                if(max < name[i]) {
                    max = name[i];
                }
            }

            bool foundSame = false;
            for(int i = 0; i < this.name.Count; i++){
                if(max < this.name[i]) return true;
                
                if(max == this.name[i])foundSame = true;
            }

            if(!foundSame) return false;

            for(; max >= 1; max--){
                int count1 = 0, count2 = 0;
                for(int i = 0; i < name.Count; i++){
                    if(max == name[i])count1++;
                }
                for(int i = 0; i < this.name.Count; i++){
                    if(max == this.name[i])count2++;
                }

                if(count1 > count2) return false;
                if(count1 < count2) return true;
            }

            return false;
        }

        List<int> smaller(int iteration){
            if(name == null) return null;
            if(name.Count == 1 && name[0] == 1) return name;
            if(name[name.Count - 1] > 1){
                name[name.Count - 1]--;
                List<int> tempName = new List<int>(name);
                name[name.Count - 1]++;
                return tempName;
            }

            List<int> r = new List<int>();

            return r;
        }

        public int getLast(){
            return name[name.Count - 1];
        }

    }

    List<Node> nodeList = new List<Node>();
    int patternLength = 0;

    public void reset(){
        nodeList.Clear();
        patternLength = 0;
    }

    int modulate(int x, int elementNumber){
        if(elementNumber < 0) return -1;
        while(x < 0) x += elementNumber;
        return (x + elementNumber) % elementNumber;
    }

    public T interpretation(T pattern){
        
        T lastChange;

        patternLength++;

        int appendNode(int startIndex, T newValue){
        
            T implication;
            
            if(startIndex < nodeList.Count){
                implication = newValue.deriveImplication(nodeList[startIndex].getValues());
                nodeList[startIndex].setValue(newValue);
                newValue = implication;

                startIndex++;
            }

            int tempIndex = startIndex;

            List<int> nameToLook = new List<int>{1};

            for(int i = startIndex; i < nodeList.Count; i++){
                if(nodeList[i].nameEquals(nameToLook)){
                    tempIndex = i + 1;

                    implication = newValue.deriveImplication(nodeList[i].getValues()); 
                    nodeList[i].setValue(newValue);
                    newValue = implication;

                }else{
                    break;
                }
            }

            if(tempIndex < nodeList.Count){
                nodeList.Insert(tempIndex, new Node(new List<int>(){1}, newValue));
            }else if(nodeList.Count > 0){
                nodeList.Add(new Node(new List<int>(){1}, newValue));
            }else{
                nodeList.Add(new Node(new List<int>(){0}, newValue));
            }

            return tempIndex;
        }

        int lastNotZeroIndex = 0;

        int tempIndex = appendNode(0, pattern);

        while(true){
                
            bool isThereSameBelow = false;
            T lastValueImplication = default(T);
            List<T> firstValues = new List<T>();
            List<int> nameToLook = nodeList[tempIndex].getName();

            for(int i = tempIndex - 1; i >= 0; i--){
                if(nodeList[i].biggerThan(nameToLook))
                    break;
                else if(nodeList[i].nameEquals(nameToLook)){
                    isThereSameBelow = true;
                    firstValues.Insert(0, nodeList[i].getFirstValue());
                    //lastValueDifference = modulate(nodeList[tempIndex].getFirstValue() - nodeList[i].getFirstValue(), elementNumber);
                    //break;
                }
            }

            if(isThereSameBelow) lastValueImplication = nodeList[tempIndex].getFirstValue().deriveImplication(firstValues);

            if(!isThereSameBelow) {
                if(!nodeList[tempIndex].getHasBigger()){
                    lastChange = nodeList[tempIndex].getValue();
                    break;
                }

                nameToLook = nodeList[tempIndex].getLesserFromBigger();
                
                bool isLesserFromBiggerThere = false;
                int lesserIndex = tempIndex;
                firstValues.Clear();

                while(true){
                    
                    for(int i = lesserIndex - 1; i >= 0; i--){
                        if(nodeList[i].biggerThan(nameToLook))
                            break;
                        else if(nodeList[i].nameEquals(nameToLook)){
                            isLesserFromBiggerThere = true;
                            lesserIndex = i;
                        }
                    }

                    if(!isLesserFromBiggerThere) break;

                    if(lesserIndex == tempIndex) break;

                    firstValues.Insert(0, nodeList[lesserIndex].getFirstValue());
                    nameToLook = nodeList[lesserIndex].getLesserFromBigger();

                    if(!nodeList[lesserIndex].getHasBigger()) break;

                    if(nameToLook.Count() <= 1){
                        if(nameToLook[0] <= 1) break;
                    }
                }

                if(isLesserFromBiggerThere)lastValueImplication = nodeList[tempIndex].getFirstValue().deriveImplication(firstValues);

                nameToLook = nodeList[tempIndex].getBigger();
            }else{
                nameToLook = nodeList[tempIndex].getIncremented();
            }
                
            bool isThereAlready = false;

            int indexToPut = tempIndex + 1;
            for(; indexToPut < nodeList.Count; indexToPut++){
                if(nodeList[indexToPut].biggerThan(nameToLook))
                    break;
                else if(nodeList[indexToPut].nameEquals(nameToLook)){
                    isThereAlready = true;
                    break;
                }    
            }

            if(isThereAlready){
                tempIndex = appendNode(indexToPut, lastValueImplication);
            }else{

                if(indexToPut < nodeList.Count){
                    nodeList.Insert(indexToPut, new Node(nameToLook, lastValueImplication));
                }else{
                    nodeList.Add(new Node(nameToLook, lastValueImplication));
                }
                
                tempIndex = indexToPut;
            }
        }

        
        if(patternLength <= 1) lastChange = default(T);
        return lastChange;
    }

    /*
    public int interpretation_unused(int elementNumber, int pattern){
        
        int lastChange = 0;

        patternLength++;

        int appendNode(int startIndex, int newValue){
        
            int difference = 0;
            
            if(startIndex < nodeList.Count){
                difference = modulate(newValue - nodeList[startIndex].getValue(), elementNumber); 
                nodeList[startIndex].setValue(newValue);
                newValue = difference;

                startIndex++;
            }

            int tempIndex = startIndex;

            for(int i = startIndex; i < nodeList.Count; i++){
                if(nodeList[i].getLast() == 1){
                    tempIndex = i + 1;

                    difference = modulate(newValue - nodeList[i].getValue(), elementNumber); 
                    nodeList[i].setValue(newValue);
                    newValue = difference;

                }else{
                    break;
                }
            }

            if(tempIndex < nodeList.Count){
                nodeList.Insert(tempIndex, new Node(new List<int>(){1}, newValue));
            }else if(nodeList.Count > 0){
                nodeList.Add(new Node(new List<int>(){1}, newValue));
            }else{
                nodeList.Add(new Node(new List<int>(){0}, newValue));
            }

            return tempIndex;
        }

        int lastNotZeroIndex = 0;

        int tempIndex = appendNode(0, modulate(pattern, elementNumber));

        while(true){
                
            bool isThereSameBelow = false;
            int nameToLook = nodeList[tempIndex].getLast(), lastValueDifference = 0;

            for(int i = tempIndex - 1; i >= 0; i--){
                if(nameToLook < nodeList[i].getLast())
                    break;
                else if(nameToLook == nodeList[i].getLast()){
                    isThereSameBelow = true;
                    lastValueDifference = modulate(nodeList[tempIndex].getFirstValue() - nodeList[i].getFirstValue(), elementNumber);
                    break;
                }
            }

            if(nodeList[tempIndex].getValue() != 0)lastNotZeroIndex = tempIndex;

            if(!isThereSameBelow) {
                lastChange = nodeList[tempIndex].getValue();
                break;
            }
            
            bool isThereAlready = false;
            nameToLook++;

            int indexToPut = tempIndex + 1;
            for(; indexToPut < nodeList.Count; indexToPut++){
                if(nameToLook < nodeList[indexToPut].getLast())
                    break;
                else if(nameToLook == nodeList[indexToPut].getLast()){
                    isThereAlready = true;
                    break;
                }    
            }

            if(isThereAlready){
                tempIndex = appendNode(indexToPut, lastValueDifference);
            }else{

                if(indexToPut < nodeList.Count){
                    nodeList.Insert(indexToPut, new Node(new List<int>(){nameToLook}, lastValueDifference));
                }else{
                    nodeList.Add(new Node(new List<int>(){nameToLook}, lastValueDifference));
                }
                
                tempIndex = indexToPut;
            }
        }

        
        if(patternLength <= 1) lastChange = 0;
        return lastChange;
    }
    */

    /*
    public PatternReader copy(){
        PatternReader r = new PatternReader();
        r.setNodeList(getNodeListCopy());
        r.setPatternLength(patternLength);
        return r;
    }
    */

    /*
    public T predict(bool continueExpected){
        List<Node> tempNodeList = getNodeListCopy();
        int tempPatternLength = patternLength;
        int lastChange = interpretation(0);
        T prediction = lastChange;
        
        nodeList.Clear();
        nodeList = tempNodeList;
        patternLength = tempPatternLength;
        
        if(continueExpected){
            interpretation(prediction);
        }
        
        return prediction;
    }
    */
    
    /*
    public float score(){
        int maxName = 1;
        float weight = 1, maxScore = 0, score = 0;

        for(int i = 0; i < nodeList.Count; i++){
            weight = 1;
            if(nodeList[i].getFirstValue() != 0){
                score += weight;
            }
            maxScore += weight;
        }

        return score/maxScore;
    }
    */

    public string toStringNodeList(bool simplify = true){
        string str = "";
        int oneLengthCounter = 0;
        for(int i = 0; i < nodeList.Count; i++){
            if(nodeList[i].getName().Count > 1 || !simplify){
                if(simplify) str += oneLengthCounter + " ";
                str += "[" + nodeList[i].nameToString() + ", " + nodeList[i].valuesToString() + "] \n";
                oneLengthCounter = 0;
            }else oneLengthCounter++;
        }

        if(oneLengthCounter > 0) str += oneLengthCounter + " ";

        return str;
    }

    public int getPatternLength(){
        return patternLength;
    }

    public int getNodeListCount(){
        return nodeList.Count;
    }

    public void setPatternLength(int patternLength){
        this.patternLength = patternLength;
    }

    public List<Node> getNodeListCopy(){
        List<Node> tempNodeList = new List<Node>();
        for(int i = 0; i < nodeList.Count; i++){
            
            tempNodeList.Add(new Node(nodeList[i].copyName(), nodeList[i].getValue()));
            tempNodeList[i].setFirstValue(nodeList[i].getFirstValue());
        }
        return tempNodeList;
    }

    public void setNodeList(List<Node> nodeList){
        this.nodeList = nodeList;
    }
}
