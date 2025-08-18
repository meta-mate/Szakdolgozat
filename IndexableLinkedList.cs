using System;
using System.Collections.Generic;

public class IndexableLinkedList<T>
{
    private readonly LinkedList<T> _list = new LinkedList<T>();

    private LinkedListNode<T> _cachedNode = null;
    private int _cachedIndex = -1;

    public int Count => _list.Count;

    public LinkedListNode<T> this[int index] => GetNodeAt(index);

    public LinkedListNode<T> First => _list.First;
    public LinkedListNode<T> Last => _list.Last;

    private LinkedListNode<T> GetNodeAt(int index)
    {
        if (index < 0 || index >= _list.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        // Use cached traversal if helpful
        if (_cachedNode != null)
        {
            while (_cachedIndex < index && _cachedNode.Next != null)
            {
                _cachedNode = _cachedNode.Next;
                _cachedIndex++;
            }

            while (_cachedIndex > index && _cachedNode.Previous != null)
            {
                _cachedNode = _cachedNode.Previous;
                _cachedIndex--;
            }

            if (_cachedIndex == index)
                return _cachedNode;
        }

        // Fall back to fresh traversal
        if (index < _list.Count / 2)
        {
            var node = _list.First;
            int i = 0;
            while (i < index)
            {
                node = node.Next;
                i++;
            }
            _cachedNode = node;
            _cachedIndex = index;
            return node;
        }
        else
        {
            var node = _list.Last;
            int i = _list.Count - 1;
            while (i > index)
            {
                node = node.Previous;
                i--;
            }
            _cachedNode = node;
            _cachedIndex = index;
            return node;
        }
    }


    public void Add(T value)
    {
        _list.AddLast(value);
        InvalidateCache();
    }

    public void AddLast(T value)
    {
        _list.AddLast(value);
        InvalidateCache();
    }

    public void AddFirst(T value)
    {
        _list.AddFirst(value);
        InvalidateCache();
    }

    public void Insert(int index, T item)
    {
        if (index < 0 || index > _list.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        if (index == _list.Count)
        {
            _list.AddLast(item);
            _cachedNode = _list.Last;
            _cachedIndex = index;
            return;
        }

        // Reuse cached node if possible
        LinkedListNode<T> node = GetNodeAt(index);
        _list.AddBefore(node, item);

        // If insertion was before cached node, shift the cached index
        if (_cachedNode != null && _cachedIndex >= index)
        {
            _cachedIndex++;
        }
    }

    public bool Remove(T value)
    {
        bool removed = _list.Remove(value);
        if (removed)
            InvalidateCache();
        return removed;
    }

    public void RemoveAt(int index)
    {
        if (index < 0 || index >= _list.Count)
            throw new ArgumentOutOfRangeException(nameof(index));

        var node = GetNodeAt(index);
        _list.Remove(node);

        // Update cache if necessary
        if (_cachedNode == node)
        {
            // Move cache to next node if possible, otherwise invalidate
            if (node.Next != null)
            {
                _cachedNode = node.Next;
                _cachedIndex = index;
            }
            else if (node.Previous != null)
            {
                _cachedNode = node.Previous;
                _cachedIndex = index - 1;
            }
            else
            {
                InvalidateCache();
            }
        }
        else if (_cachedNode != null && _cachedIndex > index)
        {
            // Adjust cached index if it was after the removed node
            _cachedIndex--;
        }
    }


    public void Clear()
    {
        _list.Clear();
        InvalidateCache();
    }

    private void InvalidateCache()
    {
        _cachedNode = null;
        _cachedIndex = -1;
    }
}
