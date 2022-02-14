package structure;

import weka.core.Instance;
import weka.core.Instances;

public interface Tree {
    void buildTree(Instances data);
    void findKNearestNeighbours(Instance instance, int k);

}
