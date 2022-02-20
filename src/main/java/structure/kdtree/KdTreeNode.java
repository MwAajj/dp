package structure.kdtree;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import weka.core.Instance;

import java.io.Serializable;

@Getter
@Setter
@NoArgsConstructor
public class KdTreeNode implements Serializable {
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
