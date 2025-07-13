using System.Collections;
using System.Collections.Generic;


public interface NodeValue<T> where T : NodeValue<T> {
    T deriveImplication<T>(IndexableLinkedList<T> nodeValues);
    T createEmpty();
    T copy();
    string ToString();
}

public class PatternReader<T> where T : NodeValue<T>
{

    class NameValue : NodeValue<NameValue>
    {

        bool value;
        public NameValue(bool value)
        {
            this.value = value;
        }

        public NameValue deriveImplication<NameValue>(IndexableLinkedList<NameValue> lesserNodes)
        {
            return createEmpty();
        }

        public NameValue createEmpty()
        {
            return new NameValue(false);
        }

        public bool getValue() => value;

        public void setValue(bool value) => this.value = value;

        public NameValue copy() => new NameValue(value);

        string ToString()
        {
            return value ? "1" : "0";
        }

    }

    public class Name
    {
        int simpleValue;
        PatternReader<NameValue> patternReader;
        Name incremented, bigger, lesserFromBigger;

        public Name(int simpleValue)
        {
            this.simpleValue = simpleValue;
        }

        public Name(PatternReader patternReader)
        {
            this.patternReader = patternReader;
        }

        public bool isSimple()
        {
            return patternReader == null;
        }

        public PatternReader<T>.Node this[int i] => patternReader.getNodeList()[i];

        public Name copy()
        {
            if (isSimple())
            {
                return new Name(simpleValue);
            }
            else
            {
                return new Name(patternReader.copy());
            }
        }

        public void increment()
        {
            if (isSimple())
            {
                simpleValue++;
            }
            else
            {
                patternReader.interpretation(new NameValue(true));
            }
        }

        public static bool operator ==(Name name, Name otherName)
        {

            if (name.isSimple() != otherName.isSimple())
            {
                return false;
            }

            if (name.isSimple() && otherName.isSimple())
            {
                return name.getSimpleValue() == otherName.getSimpleValue();
            }

            if (name.count() != otherName.count())
            {
                return false;
            }

            for (int i = 0; i < name.count(); i++)
            {
                for (int j = 0; j < name[i].getValues().Count; j++)
                {
                    bool value1 = name[i].getValues()[j].getValue();
                    bool value2 = otherName[i].getValues()[j].getValue();


                    if (value1 != value2)
                    {
                        return false;
                    }
                }
            }

            return true;

        }

        public static bool operator !=(Name name, Name otherName)
        {
            return !(name == otherName);
        }

        public Name copy(Name copiedName = null)
        {
            Name name = this;

            if (name.isSimple())
            {
                return name.getSimpleValue();
            }

            if (copiedName == null)
            {
                copiedName = new Name(new PatternReader<NameValue>());

                for (int i = 0; i < name.count(); i++)
                {
                    copiedName.increment();
                }
            }

            for (int i = 0; i < copiedName.count(); i++)
            {
                int i_adapted = i;

                while (i_adapted < name.count() && !name[i_adapted].name == copiedName[i].getName())
                    i_adapted++;

                for (int j = 0; j < copiedName[i].getValues().Count; j++)
                {
                    bool value = name[i_adapted].getValues()[j].getValue();
                    copiedName[i].getValues()[j].getValue() = value;
                }
            }

            return copiedName;

        }

        Name calculateIncremented()
        {
            incremented = copy();

            if (incremented.isSimple())
            {
                incremented.setSimpleValue(name.getSimpleValue() + 1);
            }
            else
            {
                incremented.increment();
            }

            this.incremented = incremented;
            return incremented;
        }

        Name calculateBigger()
        {
            Name name = this;

            hasBigger = false;

            if (name.isSimple())
            {
                if (name.getSimpleValue() > 1)
                {

                    Name bigger = new Name(new PatternReader<NameValue>());

                    for (int i = 0; i < 2; i++)
                    {
                        bigger.increment();
                    }

                    bigger[1].getValues()[0] = true;

                    hasBigger = true;

                    this.bigger = bigger;
                    return bigger;

                }
                else
                {
                    return null;
                }
            }
            else
            {
                Name bigger = Name(new PatternReader<NameValue>());

                int biggerSize = 0;
                for (int i = 1; i < name.count(); i++)
                {
                    for (int j = 0; j < name[i].getValues(); j++)
                    {
                        bool value = name[i].getValues()[j].getValue();
                        bool lesserValue = name[i].getLesserValue(j + 1);

                        if (lesserValue)
                        {
                            if (!value)
                            {
                                biggerSize = name[i].getFirstOccurences()[j];

                                if (i == 0)
                                {
                                    Name template = Name(new PatternReader<NameValue>());
                                    for (int k = 0; k < name[0].getValues().Count - 1; k++)
                                    {
                                        template.increment();
                                    }
                                    lesserFromBigger = copy(template);
                                }
                                else
                                {
                                    lesserFromBigger = copy();

                                    for (int k = 0; k < name[0].getLesserLength(); k++)
                                    {
                                        bool condition = k + 1 >= name[0].getLesserLength();
                                        if (!condition)
                                            condition = !name[i].getLesserValue(k + 1);

                                        if (condition)
                                        {
                                            lesserFromBigger[i].getLesserValue(k) = false;
                                            break;
                                        }
                                    }

                                }

                                break;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                }

                for (int i = 0; i < biggerSize; i++)
                {
                    bigger.increment();
                }

                bigger = copy(bigger);

                for (int i = 1; i < bigger.count(); i++)
                {
                    for (int j = 0; j < bigger[i].getValues(); j++)
                    {
                        bool value = bigger[i].getValues()[j].getValue();
                        bool lesserValue = bigger[i].getLesserValue(j + 1);

                        if (lesserValue)
                        {
                            if (!value)
                            {
                                bigger[i].getValues()[j].setValue(true);
                                hasBigger = true;
                                break;
                            }
                        }
                    }
                    if (hasBigger)
                    {
                        break;
                    }
                }

                this.bigger = bigger;
                return this.bigger;
            }
        }

        public static bool operator >(Name name, Name otherName)
        {

            if (name.isSimple() && otherName.isSimple())
            {
                return name.getSimpleValue() > otherName.getSimpleValue();
            }
            else if (!name.isSimple() && otherName.isSimple())
            {
                return true;
            }
            else if (name.isSimple() && !otherName.isSimple())
            {
                return false;
            }

            Name longerName = name;
            Name shortherName = otherName;

            bool isShorter = name.count() < otherName.count();

            if (isShorter)
            {
                longerName = otherName;
                shortherName = name;
            }

            int isShorerBigger = 0;

            for (int i = 0; i < shortherName.count(); i++)
            {
                int i_adapted = i;

                while (i_adapted < longerName.count() && !shortherName[i].name == longerName[i_adapted].getName())
                    i_adapted++;

                IndexableLinkedList<NameValue> shorterValues = shortherName[i].getValues();
                IndexableLinkedList<NameValue> longerValues = longerName[i_adapted].getValues();

                for (int j = 0; j < longerValues.Count; j++)
                {
                    bool shorterValue = false;
                    bool longerValue = longerValues[j].getValue();

                    if (j < shorterValues.count())
                        shorterValue = shorterValues[j].getValue();

                    if (shorterValue && !longerValue)
                    {
                        isShorerBigger = 1;
                        break;
                    }
                    else if (!shorterValue && longerValue)
                    {
                        isShorerBigger = -1;
                        break;
                    }
                }
            }

            if (isShorer)
            {
                return isShorerBigger > 0;
            }
            else
            {
                return isShorerBigger > 0;
            }
        }

        public static bool operator <(Name name, Name otherName)
        {
            return otherName > name;
        }

        public int getSimpleValue() => simpleValue;

        public void setSimpleValue(int simpleValue) => this.simpleValue = simpleValue;

        public PatternReader getPatternReader() => patternReader;

        public Name getName() => name;

        public Name getIncremented()
        {
            if (incremented == null) calculateIncremented();
            return incremented;
        }

        public Name getHasBigger()
        {
            if (bigger == null) calculateBigger();
            return hasBigger;
        }

        public Name getBigger()
        {
            if (bigger == null) calculateBigger();
            return bigger;
        }

        public Name getLesserFromBigger()
        {
            if (bigger == null) calculateBigger();
            return lesserFromBigger;
        }

        public int count()
        {
            if (isSimple())
            {
                return Math.Min(1, simpleValue);
            }
            else
            {
                return patternReader.getNodeList().Count;
            }
        }

        public string ToString()
        {
            if (isSimple())
            {
                return simpleValue.ToString();
            }
            else
            {
                return patternReader.ToString();
            }
        }

    }

    public class Node
    {
        Name name;
        IndexableLinkedList<T> values;
        IndexableLinkedList<Node> lesserNodes;
        IndexableLinkedList<int> firstOccurences;
        bool hasBigger = false;

        public Node(Name name, IndexableLinkedList<Node> lesserNodes = null)
        {
            this.name = name;
            values = new IndexableLinkedList<T>();


            if (lesserNodes != null)
            {
                this.lesserNodes = lesserNodes;
            }
            else
            {
                this.lesserNodes = new IndexableLinkedList<Node>();
            }


        }

        public IndexableLinkedList<T> getValues()
        {
            return values;
        }

        public void appendValues(T value, int patternLength)
        {
            values.Add(value);
            firstOccurences.Add(patternLength);
        }

        public T getLesserValue(int index)
        {
            if (name == Name(1))
                return lesserNodes[0].getValues()[index];
            else
                return lesserNodes[index].getValues()[0];
        }

        public int getLesserLength()
        {
            return values.count + 1;
        }

        public IndexableLinkedList<T> getLesserNodes()
        {
            return lesserNodes;
        }

        public IndexableLinkedList<int> getFirstOccurences()
        {
            return firstOccurences;
        }

        public Name getName()
        {
            return name;
        }

        public void setName(Name name)
        {
            this.name = name;
        }

        public string ToString()
        {
            string valuesStr = "";

            for (int i = 0; i < values.Count; i++)
            {
                valuesStr += values[i];
                if (i < values.Count - 1) valuesStr += " | ";
            }

            return "Node(Name: " + name + " Values: " + valuesStr + ")";
        }

    }

    IndexableLinkedList<Node> nodeList = new IndexableLinkedList<Node>();
    int patternLength = 0;

    public void reset()
    {
        nodeList.Clear();
        patternLength = 0;
    }

    public T interpretation(T pattern)
    {

        T lastChange = default(T);

        patternLength++;

        int appendNode(int startIndex, T lesserNode = null)
        {

            IndexableLinkedList<Node> lesser_nodes = null;

            if (startIndex < nodeList.Count)
            {

                nodeList[startIndex].getLesserNodes().Add(lesserNode);
                nodeList[startIndex].appendValues(pattern.createEmpty(), patternLength);

                startIndex++;
            }

            int tempIndex = startIndex;

            Name nameOne = new Name(1);

            Name nameZero = new Name(0);

            Name nameToLook = nameOne;

            for (int i = startIndex; i < nodeList.Count; i++)
            {
                if (nodeList[i].getName() == nameToLook)
                {
                    tempIndex = i + 1;

                    nodeList[i].appendValues(pattern.createEmpty(), patternLength);

                }
                else
                {
                    break;
                }
            }

            if (tempIndex > 0)
            {
                lesser_nodes.Add(nodeList[tempIndex - 1]);
            }

            if (tempIndex < nodeList.Count)
            {
                nodeList.Insert(tempIndex, new Node(nameOne, lesser_nodes));
                nodeList[tempIndex].appendValues(pattern.createEmpty(), patternLength);
            }
            else if (nodeList.Count > 0)
            {
                nodeList.Add(new Node(nameOne, lesser_nodes));
                nodeList[tempIndex].appendValues(pattern.createEmpty(), patternLength);
            }
            else
            {
                nodeList.Insert(tempIndex, new Node(nameZero, lesser_nodes));
                nodeList[tempIndex].appendValues(pattern.createEmpty(), patternLength);
            }

            return tempIndex;
        }

        int tempIndex = appendNode(0);
        nodeList[0].getValues().Last = pattern;

        while (true)
        {

            bool isThereSameBelow = false;

            IndexableLinkedList<T> lesser_nodes = new IndexableLinkedList<T>();
            Name nameToLook = nodeList[tempIndex].getName();

            for (int i = tempIndex - 1; i >= 0; i--)
            {
                if (nodeList[i].getName() == nameToLook)
                {
                    isThereSameBelow = true;
                    lesser_nodes.AddFirst(nodeList[i]);
                    break;
                }
                if (nodeList[i].getName() > nameToLook)
                    break;
            }


            if (!isThereSameBelow)
            {
                if (!nameToLook.getHasBigger())
                {
                    lastChange = nodeList[tempIndex].getValues().Last;
                    break;
                }

                //nameToLook = nodeList[tempIndex].getName();
                Name lesserName = nameToLook.getName();
                Name bigger = nameToLook.getBigger();

                bool isLesserFromBiggerThere = false;
                int lesserIndex = tempIndex;
                lesser_nodes.Clear();

                for (int i = lesserIndex - 1; i >= 0; i--)
                {
                    if (nodeList[i].getName() == bigger)
                        break;
                    if (nodeList[i].getName() > bigger)
                        break;

                    if (nodeList[i].getName() == nameToLook)
                    {
                        isLesserFromBiggerThere = false;
                        break;
                    }
                    if (nodeList[i].getName() > nameToLook)
                    {
                        isLesserFromBiggerThere = false;
                        break;
                    }

                    if (nodeList[i].getName() == nameToLook)
                    {
                        isLesserFromBiggerThere = true;
                        lesserIndex = i;
                    }
                }

                if (!isLesserFromBiggerThere)
                    break;

                lesser_nodes.AddFirst(nodeList[lesserIndex]);

                nameToLook = nameToLook.getBigger();
            }
            else
            {
                nameToLook = nameToLook.getIncremented();
            }

            bool isThereAlready = false;

            int indexToPut = tempIndex + 1;
            while (indexToPut < nodeList.Count)
            {
                if (nodeList[indexToPut].name == nameToLook)
                {
                    isThereAlready = true;
                    break;
                }
                if (nodeList[indexToPut].name > nameToLook)
                    break;

                indexToPut++;
            }

            lesser_nodes.Add(nodeList[tempIndex]);

            if (isThereAlready)
            {
                tempIndex = appendNode(indexToPut, lesser_nodes.Last);
            }
            else
            {

                if (indexToPut < nodeList.Count)
                {
                    nodeList.Insert(indexToPut, new Node(nameToLook, lesser_nodes));
                }
                else
                {
                    nodeList.Add(new Node(nameToLook, lesser_nodes));
                }
                nodeList[indexToPut].appendValues(pattern.createEmpty(), patternLength);

                tempIndex = indexToPut;
            }
        }

        return lastChange;
    }

    public void calculateValues()
    {
        for (int i = 1; i < nodeList.Count; i++)
        {
            IndexableLinkedList<T> values = nodeList[i].getValues();
            IndexableLinkedList<T> lesserValues = new IndexableLinkedList<T>();
            int lesserLength = getLesserLength();
            for (int j = 0; j < lesserLength; j++)
            {
                lesserValues.Add(getLesserValue(j));
            }
            for (int j = 0; j < values.Count; j++)
            {
                values[j].deriveImplication(lesserValues, j);
            }
        }
    }

    public string ToString()
    {

        string str = "";

        for (int i = 0; i < nodeList.Count; i++)
        {
            str += nodeList[i].toString();
        }

        return str;
    }

    public int getPatternLength()
    {
        return patternLength;
    }

    public void setPatternLength(int patternLength)
    {
        this.patternLength = patternLength;
    }

    public IndexableLinkedList<PatternReader<T>.Node> getNodeList()
    {
        return nodeList;
    }

    public PatternReader<T> copy()
    {
        PatternReader<T> copy = new PatternReader<T>();

        for (int i = 0; i < patternLength; i++)
        {
            copy.interpretation(default(T));
        }

        for (int i = 0; i < nodeList.Count; i++)
        {
            copy.nodeList[i].setName(nodeList[i].getName().copy());

            IndexableLinkedList<T> values = nodeList[i].getValues();
            IndexableLinkedList<T> copy_values = copy.nodeList[i].getValues();

            for (int j = 0; j < values.Count; j++)
            {
                copy_values[j] = values[j].copy();
            }
        }

        return copy;
    }

    public void setNodeList(IndexableLinkedList<Node> nodeList)
    {
        this.nodeList = nodeList;
    }
}
