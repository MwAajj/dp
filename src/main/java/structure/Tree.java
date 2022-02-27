package structure;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.ArrayList;

public interface Tree {
    enum Son {
        NONE,
        LEFT,
        RIGHT,
        BOTH
    }
    void buildTree(Instances data);
    Instances findKNearestNeighbours(Instance instance, int k);
    ArrayList<Attribute> getALlAttributes(Instance instance);
    double[] getDistances();
}
