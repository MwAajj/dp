package structure.kdtree;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import weka.core.Instance;

@Getter
@Setter
@NoArgsConstructor
public class KdTreeNode {
    private KdTreeNode parent;
    private KdTreeNode leftSon;
    private KdTreeNode rightSon;
    private Instance instance;
    private int level;


    public KdTreeNode(Instance instance) {
        this.instance = instance;
    }


    public boolean isLeaf() {
        return this.leftSon == null && this.rightSon == null;
    }
}
