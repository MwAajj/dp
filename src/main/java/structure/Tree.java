package structure;

import weka.core.Instance;
import weka.core.Instances;

public interface Tree {
    void buildTree(Instances data);
    Instances findKNearestNeighbours(Instance instance, int k);

}
