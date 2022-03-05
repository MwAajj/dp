package structure.trees.kdtree;

import lombok.Getter;
import lombok.Setter;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;

@Getter
@Setter
public class KdTreeNode implements Serializable {
    private KdTreeNode leftSon;
    private KdTreeNode rightSon;
    private Instance instance;
    private int level;
    private Instances instances;


    public KdTreeNode(Instance instance) {
        this.instance = instance;
    }

    public boolean isInstances() {
        return instances != null;
    }

    public boolean isLeaf() {
        return this.leftSon == null && this.rightSon == null;
    }

    public void clearInstances() {
        instances = null;
    }
}
