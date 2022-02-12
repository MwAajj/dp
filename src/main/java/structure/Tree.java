package structure;

import weka.core.Instance;
import weka.core.Instances;

public interface Tree {
    void buildTree(Instances data);
    double classifyInstance(Instance instance, int k);
}
