import lombok.Getter;
import lombok.Setter;
import structure.Tree;
import structure.balltree.BallTree;
import structure.kdtree.KdTree;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.*;

import java.util.Enumeration;

@Getter
@Setter
public class MyAlgorithm extends AbstractClassifier implements Classifier, OptionHandler {

    private boolean mk_isKdTree = false;
    private boolean mk_isBallTree = false;
    private boolean mk_noWeight = false;
    private boolean mk_harmonicWeight = false;

    private int k;
    private Tree tree;

    public MyAlgorithm() {
        k = 1;
    }

    public MyAlgorithm(int k) {
        this.k = k;
    }

    // faza trenovavania instances
    //
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (mk_isBallTree)
            tree = new BallTree();
        else if (mk_isKdTree)
            tree = new KdTree();
        tree.buildTree(data);
    }

    //vratit nieco ako index cielovej triedy
    @Override
    public double classifyInstance(Instance instance) throws Exception {
       return tree.classifyInstance(instance, k);
    }


    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
    }

    //kedy je ten algoritmus pouzitelny,
    // ked je to binarny klasifikator a su trie triedy exception
    // numeralne hodnoty vs lingvisticke
    @Override
    public Capabilities getCapabilities() {
        return null;
    }

    @Override
    public Enumeration<Option> listOptions() {
        return null;
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        mk_isKdTree = Utils.getFlag('K', options);
        mk_isBallTree = Utils.getFlag('B', options);
        mk_noWeight = Utils.getFlag('N', options);
        mk_harmonicWeight = Utils.getFlag('H', options);
    }

    @Override
    public String[] getOptions() {
        return new String[0];
    }
}
