package structure;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;

public interface Tree {
    void buildTree(Instances data);
    Instances findKNearestNeighbours(Instance instance, int k);
    ArrayList<Attribute> getALlAttributes(Instance instance);
}
