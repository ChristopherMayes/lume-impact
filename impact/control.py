import json


class ControlGroup:
    """
    Group elements to control a change in an attribute for a list of elements. 
    
    Based on Bmad's GROUP element. 
    
    
    Example:
        ELES = {'a':{'x':1}, 'b':{'x':2}}
        G = ControlGroup(ele_names=['a', 'b'], attribute='x')
        G.link(ELES)
        G['x'] = 5
        
        will make ELES = {'a': {'x': 6}, 'b': {'x': 7}}
    
    """
    def __init__(self, ele_names=[], attribute=None, value=0, old_value=0):
        
        self.ele_names = ele_names # Link these. 
        self.attribute = attribute
        self.value = value
        self.old_value = old_value
        
        # These need to be linked
        self.eles = None
    
    def link(self, eles):
        """
        Link and ele dict, so that update will work
        """
        self.eles=eles
        self.update()

    def update(self):
        """
        Updates linked eles with any change in the group value
        """
        assert self.eles, 'No eles are linked. Please call .link(eles)'
        
        dval = self.value - self.old_value
        for name in self.ele_names:
            self.eles[name][self.attribute] += dval
        self.old_value = self.value
    
    def __setitem__(self, key, item):
        assert key == self.attribute
        self.value = item
        self.update()
        
    def __getitem__(self, key):
        assert key == self.attribute
        return self.value
    
    def __str__(self):
        return f'Group of eles {[e for e in self.eles]} with delta {self.attribute} = {self.value}'
    
    
    def dumps(self):
        """
        Dump the internal data as a JSON string
        """
        eles = self.__dict__.pop('eles')
        d = self.__dict__
        s = json.dumps(d)
        # Relink
        self.eles = eles
        return s
    
    def loads(self, s):
        """
        Loads from a JSON string. See .dumps()
        """
        d = json.loads(s)
        self.__dict__.update(d)
    
    def __repr__(self):
        
        s0 = self.dumps()
        s = f'ControlGroup(**{s0})'
        
        return s
        