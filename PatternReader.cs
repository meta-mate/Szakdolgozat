using System.Collections;
using System.Collections.Generic;

public abstract class NodeValue<T1>
{

    public T1 Value { get; set; }
    public bool IsEmpty { get; set; }

    public NodeValue(T1 value, bool isEmpty = false)
    {
        Value = value;
        IsEmpty = isEmpty;
    }
    public abstract NodeValue<T1> calculateValue(IndexableLinkedList<NodeValue<T1>> lesserNodes, int index);

    public bool calculateIfNeeded(IndexableLinkedList<NodeValue<T1>> lesserNodes, int index)
    {
        if (IsEmpty)
        {
            IsEmpty = false;
            calculateValue(lesserNodes, index);
            return true;
        }
        else
        {
            return false;
        }
    }

    public abstract NodeValue<T1> createEmpty();
    public abstract NodeValue<T1> copy();
    public override string ToString()
    {
        return Value.ToString();
    }
}

public class NameValue : NodeValue<bool>
{

    public NameValue(bool value) : base(value) { }

    public override NodeValue<bool> calculateValue(IndexableLinkedList<NodeValue<bool>> lesserNodes, int index)
    {
        return this;
    }

    public override NodeValue<bool> createEmpty()
    {
        return new NameValue(false);
    }

    public override NodeValue<bool> copy() => new NameValue(Value);

    public override string ToString()
    {
        return Value ? "1" : "0";
    }

}

public class Name
{
    public int SimpleValue { get; set; }
    public PatternReader<bool> PatternReader { get; set; }
    public Name Incremented { get; set; }
    public Name Greater { get; set; }
    public Name LesserFromGreater { get; set; }
    public bool HasGreater { get; set; }

    private bool isGreaterCalculated = false;

    public Name(int simpleValue)
    {
        SimpleValue = simpleValue;
        PatternReader = null;

        isGreaterCalculated = false;
        Greater = null;
        LesserFromGreater = null;
        HasGreater = false;
    }

    public Name(PatternReader<bool> patternReader)
    {
        PatternReader = patternReader;
    }

    public bool isSimple()
    {
        return PatternReader == null;
    }

    public Name simplify()
    {
        if (isSimple()) return this;

        if (PatternReader.PatternLength == 2)
        {
            if (this[1].Values.First.Value.Value)
            {
                SimpleValue = 1;
                PatternReader = null;
                return this;
            }
        }

        if (PatternReader.PatternLength == 3)
        {
            for (int i = 1; i < this.Count; i++)
            {
                for (int j = 0; j < this[i].Values.Count; j++)
                {
                    if (i != 2 && j != 0)
                    {
                        if (this[i].Values[j].Value.Value)
                        {
                            return this;
                        }
                    }
                }
            }

            SimpleValue = -1;
            PatternReader = null;
            return this;
        }
        return this;
    }

    public Node<bool> this[int i] => PatternReader.NodeList[i].Value;

    public void increment()
    {
        if (isSimple())
        {
            if (SimpleValue > 0)
            {
                SimpleValue++;
            }
            else
            {
                PatternReader = new PatternReader<bool>();
                for (int i = 0; i < 3; i++)
                {
                    PatternReader.interpretation(new NameValue(false));
                }

                this[1].Values[0].Value.Value = true;
                this[2].Values[0].Value.Value = true;
            }
        }
        else
        {
            int i = 0;
            IndexableLinkedList<NodeValue<bool>> values = this[1].Values;

            while (true)
            {
                while (i >= values.Count)
                {
                    PatternReader.interpretation(new NameValue(true));
                }

                NameValue value = values[i].Value as NameValue;
                if (!value.Value)
                {
                    value.Value = true;
                    break;
                }

                i++;
            }
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
            return name.SimpleValue == otherName.SimpleValue;
        }

        if (name.Count != otherName.Count)
        {
            return false;
        }

        for (int i = 0; i < name.Count; i++)
        {
            for (int j = 0; j < name[i].Values.Count; j++)
            {
                bool value1 = name[i].Values[j].Value.Value;
                bool value2 = otherName[i].Values[j].Value.Value;


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
        if (isSimple())
        {
            return new Name(SimpleValue);
        }
        else
        {
            PatternReader<bool> copied = null;
            if (copiedName is not null)
            {
                copied = copiedName.PatternReader;
            }
            return new Name(PatternReader.copy(copied));
        }
    }

    public Name calculateGreater()
    {
        isGreaterCalculated = true;

        HasGreater = false;

        if (isSimple())
        {
            if (SimpleValue > 1)
            {
                HasGreater = true;

                Greater = new Name(-1);

                LesserFromGreater = new Name(SimpleValue - 1);

                return Greater;
            }
            else
            {
                return null;
            }
        }
        else
        {
            Greater = copy();

            int greaterSize = 0;
            for (int i = 2; i < Greater.Count; i++)
            {
                int j = 0;

                bool first_lesser = Greater[i].getLesserValue(0).Value;
                bool second_lesser = Greater[i].getLesserValue(1).Value;

                if (first_lesser && second_lesser)
                {
                    int lesser_length = Greater[i].getLesserLength();

                    for (j = 1; j < lesser_length; j++)
                    {
                        if (!Greater[i].getLesserValue(j).Value || j == lesser_length - 1)
                        {
                            Name template = null;

                            int minSize = 0;

                            for (int k = i + 1; k < Greater.Count; k++)
                            {
                                for (int l = 0; l < Greater[k].Values.Count; l++)
                                {
                                    if (Greater[k].Values[l].Value.Value)
                                    {
                                        minSize = Greater[k].FirstOccurences[l].Value;
                                    }
                                    else break;
                                }
                            }

                            int lesserFromGreaterSize = Greater[i].LesserNodes[j - 1].Value.FirstOccurences[0].Value;
                            if (lesserFromGreaterSize > minSize) minSize = lesserFromGreaterSize;

                            if (Greater.PatternReader.PatternLength > minSize)
                            {
                                template = new Name(new PatternReader<bool>());
                                for (int k = 0; k < minSize; k++)
                                {
                                    template.PatternReader.interpretation(new NameValue(false));
                                }
                            }

                            LesserFromGreater = copy(template);
                            LesserFromGreater.simplify();
                            break;
                        }
                    }
                    

                    j = 0;
                    while (true)
                    {
                        while (j >= Greater[i].Values.Count)
                        {
                            Greater.PatternReader.interpretation(new NameValue(false));
                        }

                        if (!Greater[i].Values[j].Value.Value)
                        {
                            Greater[i].Values[j].Value.Value = true;
                            greaterSize = Greater[i].FirstOccurences[j].Value;

                            int k = 0;
                            while (Greater[k] != Greater[i])
                            {
                                for (int l = 0; l < Greater[k].Values.Count; l++)
                                {
                                    Greater[k].Values[l].Value.Value = false;
                                }

                                k++;
                            }

                            if (greaterSize != Greater.PatternReader.PatternLength)
                            {
                                Name template = new Name(new PatternReader<bool>());
                                for (k = 0; k < greaterSize; k++)
                                {
                                    template.PatternReader.interpretation(new NameValue(false));
                                }
                                Greater = Greater.copy(template);
                            }

                            HasGreater = true;


                            return Greater;
                        }

                        j++;
                    }
                }
            }
        }
        return null;
    }

    public static bool operator >(Name name, Name otherName)
    {

        if (name.isSimple() && otherName.isSimple())
        {
            if (name.SimpleValue < 0 && otherName.SimpleValue >= 0)
            {
                return true;
            } else if (otherName.SimpleValue < 0) {
                return false;
            }


            return name.SimpleValue > otherName.SimpleValue;
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

        bool isShorter = name.Count < otherName.Count;

        if (isShorter)
        {
            longerName = otherName;
            shortherName = name;
        }

        int isShorterGreater = 0;

        for (int i = 0; i < shortherName.Count; i++)
        {
            int i_adapted = i;

            while (i_adapted < longerName.Count && !(shortherName[i].Name == longerName[i_adapted].Name))
                i_adapted++;

            IndexableLinkedList<NodeValue<bool>> shorterValues = shortherName[i].Values;
            IndexableLinkedList<NodeValue<bool>> longerValues = longerName[i_adapted].Values;

            for (int j = 0; j < longerValues.Count; j++)
            {
                bool shorterValue = false;
                bool longerValue = longerValues[j].Value.Value;

                if (j < shorterValues.Count)
                    shorterValue = shorterValues[j].Value.Value;

                if (shorterValue && !longerValue)
                {
                    isShorterGreater = 1;
                    break;
                }
                else if (!shorterValue && longerValue)
                {
                    isShorterGreater = -1;
                    break;
                }
            }
        }

        if (isShorter)
        {
            return isShorterGreater > 0;
        }
        else
        {
            return isShorterGreater > 0;
        }
    }

    public static bool operator <(Name name, Name otherName)
    {
        return otherName > name;
    }

    public Name getIncremented()
    {
        if (Incremented is null)
        {
            Incremented = copy();
            Incremented.increment();

        }
        return Incremented;
    }

    public bool getHasGreater()
    {
        if (!isGreaterCalculated)
        {
            calculateGreater();
        }
        return HasGreater;
    }

    public Name getGreater()
    {
        if (!isGreaterCalculated) calculateGreater();
        return Greater;
    }

    public Name getLesserFromGreater()
    {
        if (!isGreaterCalculated) calculateGreater();
        return LesserFromGreater;
    }

    public int Count
    {
        get {
            if (isSimple())
            {
                return 1;
            }
            else
            {
                return PatternReader.NodeList.Count;
            }
        }
    }

    public override string ToString()
    {
        if (isSimple())
        {
            return SimpleValue.ToString();
        }
        else
        {
            return PatternReader.ToString();
        }
    }

}

public class Node<T>
{
    public Name Name { get; set; }
    public IndexableLinkedList<NodeValue<T>> Values { get; set; }
    public IndexableLinkedList<Node<T>> LesserNodes { get; set; }
    public IndexableLinkedList<int> FirstOccurences { get; set; }

    public Node(Name name, IndexableLinkedList<Node<T>> lesserNodes = null)
    {
        Name = name;
        Values = new IndexableLinkedList<NodeValue<T>>();
        FirstOccurences = new IndexableLinkedList<int>();

        if (lesserNodes != null)
        {
            LesserNodes = lesserNodes;
        }
        else
        {
            LesserNodes = new IndexableLinkedList<Node<T>>();
        }

    }

    public void appendValues(NodeValue<T> value, int PatternLength)
    {
        Values.Add(value);
        FirstOccurences.Add(PatternLength);
    }

    public NodeValue<T> getLesserValue(int index)
    {
        if (Name == new Name(1))
            return LesserNodes[0].Value.Values[index].Value;
        else
            return LesserNodes[index].Value.Values[0].Value;
    }

    public int getLesserLength()
    {
        return Values.Count + 1;
    }

    public override string ToString()
    {
        string valuesStr = "";

        for (int i = 0; i < Values.Count; i++)
        {
            valuesStr += Values[i].Value;
            if (i < Values.Count - 1) valuesStr += " | ";
        }

        return "Node(Name: " + Name.ToString() + " Values: " + valuesStr + ")";
    }

}

public class PatternReader<T>
{

    public IndexableLinkedList<Node<T>> NodeList { get; set; }
    public int PatternLength { get; set; }

    public PatternReader()
    {
        NodeList = new IndexableLinkedList<Node<T>>();
        PatternLength = 0;
    }

    public void reset()
    {
        NodeList.Clear();
        PatternLength = 0;
    }

    public NodeValue<T> interpretation(NodeValue<T> pattern)
    {
        PatternLength++;

        int appendNode(int startIndex, Node<T> lesserNode = null)
        {

            IndexableLinkedList<Node<T>> lesser_nodes = new IndexableLinkedList<Node<T>>();

            if (startIndex < NodeList.Count)
            {

                NodeList[startIndex].Value.LesserNodes.Add(lesserNode);
                NodeList[startIndex].Value.appendValues(pattern.createEmpty(), PatternLength);

                startIndex++;
            }

            int tempIndex = startIndex;

            Name nameOne = new Name(1);

            Name nameZero = new Name(0);

            Name nameToLook = nameOne;

            for (int i = startIndex; i < NodeList.Count; i++)
            {
                if (NodeList[i].Value.Name == nameToLook)
                {
                    tempIndex = i + 1;

                    NodeList[i].Value.appendValues(pattern.createEmpty(), PatternLength);

                }
                else
                {
                    break;
                }
            }

            if (tempIndex > 0)
            {
                lesser_nodes.Add(NodeList[tempIndex - 1].Value);
            }

            if (tempIndex < NodeList.Count)
            {
                NodeList.Insert(tempIndex, new Node<T>(nameOne, lesser_nodes));
            }
            else if (NodeList.Count > 0)
            {
                NodeList.Add(new Node<T>(nameOne, lesser_nodes));
            }
            else
            {
                NodeList.Insert(tempIndex, new Node<T>(nameZero, lesser_nodes));
            }

            NodeList[tempIndex].Value.appendValues(pattern.createEmpty(), PatternLength);

            return tempIndex;
        }

        int tempIndex = appendNode(0);
        NodeList[0].Value.Values.Last.Value = pattern;


        NodeValue<T> lastChange;
        while (true)
        {

            lastChange = NodeList[tempIndex].Value.Values.Last.Value;

            bool isThereSameBelow = false;

            IndexableLinkedList<Node<T>> lesser_nodes = new IndexableLinkedList<Node<T>>();
            Name nameToLook = NodeList[tempIndex].Value.Name;

            for (int i = tempIndex - 1; i >= 0; i--)
            {
                if (NodeList[i].Value.Name == nameToLook)
                {
                    isThereSameBelow = true;
                    lesser_nodes.AddFirst(NodeList[i].Value);
                    break;
                }
                if (NodeList[i].Value.Name > nameToLook)
                    break;
            }


            if (!isThereSameBelow)
            {
                if (!nameToLook.getHasGreater())
                    break;

                //nameToLook = NodeList[tempIndex].Name;
                Name lesser = nameToLook.getLesserFromGreater();
                Name greater = nameToLook.getGreater();


                bool isLesserFromGreaterThere = false;
                int lesserIndex = tempIndex;
                lesser_nodes.Clear();

                for (int i = lesserIndex - 1; i >= 0; i--)
                {
                    if (NodeList[i].Value.Name == greater)
                        break;
                    if (NodeList[i].Value.Name > greater)
                        break;

                    if (NodeList[i].Value.Name == nameToLook)
                    {
                        isLesserFromGreaterThere = false;
                        break;
                    }
                    if (NodeList[i].Value.Name > nameToLook)
                    {
                        isLesserFromGreaterThere = false;
                        break;
                    }

                    if (NodeList[i].Value.Name == lesser)
                    {
                        isLesserFromGreaterThere = true;
                        lesserIndex = i;
                    }
                }

                if (!isLesserFromGreaterThere)
                    break;

                lesser_nodes.AddFirst(NodeList[lesserIndex].Value);

                nameToLook = greater;
            }
            else
            {
                nameToLook = nameToLook.getIncremented();
            }

            bool isThereAlready = false;

            int indexToPut = tempIndex + 1;
            while (indexToPut < NodeList.Count)
            {
                if (NodeList[indexToPut].Value.Name == nameToLook)
                {
                    isThereAlready = true;
                    break;
                }
                if (NodeList[indexToPut].Value.Name > nameToLook)
                    break;

                indexToPut++;
            }

            lesser_nodes.Add(NodeList[tempIndex].Value);

            if (isThereAlready)
            {
                tempIndex = appendNode(indexToPut, lesser_nodes.Last.Value);
            }
            else
            {

                if (indexToPut < NodeList.Count)
                {
                    NodeList.Insert(indexToPut, new Node<T>(nameToLook, lesser_nodes));
                }
                else
                {
                    NodeList.Add(new Node<T>(nameToLook, lesser_nodes));
                }
                NodeList[indexToPut].Value.appendValues(pattern.createEmpty(), PatternLength);

                tempIndex = indexToPut;
            }
        }

        return lastChange;
    }

    public void calculateValues()
    {
        for (int i = 1; i < NodeList.Count; i++)
        {
            IndexableLinkedList<NodeValue<T>> values = NodeList[i].Value.Values;
            IndexableLinkedList<NodeValue<T>> lesserValues = new IndexableLinkedList<NodeValue<T>>();
            int lesserLength = NodeList[i].Value.getLesserLength();
            for (int j = 0; j < lesserLength; j++)
            {
                lesserValues.Add(NodeList[i].Value.getLesserValue(j));
            }
            for (int j = values.Count - 1; j >= 0; j--)
            {
                NodeValue<T> value = values[j].Value;
                if (!value.calculateIfNeeded(lesserValues, j))
                    break;
            }
        }
    }


    public PatternReader<T> copy(PatternReader<T> copied = null)
    {

        if (copied is null)
        {
            copied = new PatternReader<T>();

            IndexableLinkedList<NodeValue<T>> values = NodeList[0].Value.Values;
            for (int i = 0; i < values.Count; i++)
            {
                copied.interpretation(values[i].Value.copy());
            }
        }

        for (int i = 0; i < copied.NodeList.Count; i++)
        {
            int i_adapted = i;
            while (i_adapted < NodeList.Count && NodeList[i_adapted].Value.Name != copied.NodeList[i].Value.Name)
            {
                i_adapted++;
            }

            for (int j = 0; j < copied.NodeList[i].Value.Values.Count; j++)
            {
                NodeValue<T> value = NodeList[i_adapted].Value.Values[j].Value.copy();
                copied.NodeList[i].Value.Values[j].Value = value;
            }
        }

        return copied;
    }

    public override string ToString()
    {

        string str = "";

        for (int i = 0; i < NodeList.Count; i++)
        {
            str += NodeList[i].Value.ToString() + "\n";
        }

        return str;
    }
}
