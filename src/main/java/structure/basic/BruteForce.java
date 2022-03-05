package structure.basic;

import structure.MathOperation;
import structure.Structure;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.neighboursearch.NearestNeighbourSearch;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class BruteForce extends NearestNeighbourSearch implements Structure {
    private Instances instances;
    private double[] distances;

    @Override
    public void buildStructure(Instances data) {
        instances = data;
    }

    private void checkData(Instance instance, int k) {
        if (instances.size() < k)
            throw new RuntimeException("K is bigger than data");
        if (instance.classIndex() != instances.classIndex())
            throw new RuntimeException("Different class indexes");
    }

    @Override
    public Instances findKNearestNeighbours(Instance target, int k) {
        checkData(target, k);
        DistInst[] inst = new DistInst[instances.size()];
        Instances result = new Instances("neighbours", getALlAttributes(instances.firstInstance()), k);
        for (int i = 0; i < instances.size(); i++) {
            double distance = MathOperation.euclidDistance(target.classIndex(), instances.get(i), target);
            inst[i] = new DistInst(instances.get(i), distance);
        }
        Arrays.sort(inst, Collections.reverseOrder());
        for (int i = 0; i < k; i++) {
            result.add(inst[i].getInstance());
        }
        return result;
    }

    @Override
    public ArrayList<Attribute> getALlAttributes(Instance instance) {
        ArrayList<Attribute> attr = new ArrayList<>();
        for (int i = 0; i < instance.numAttributes(); i++) {
            attr.add(instance.attribute(i));
        }
        return attr;
    }

    @Override
    public Instance nearestNeighbour(Instance target) {
        if (target.classIndex() != instances.classIndex())
            throw new RuntimeException("Different class indexes");
        double min = Double.MAX_VALUE;
        Instance returnInstance = null;
        distances = new double[1];
        for (Instance instance : instances) {
            double distance = MathOperation.euclidDistance(target.classIndex(), instance, target);
            if (distance < min) {
                min = distance;
                returnInstance = instance;
                distances[0] = distance;
            }
        }
        return returnInstance;
    }

    @Override
    public Instances kNearestNeighbours(Instance target, int k) {
        return findKNearestNeighbours(target, k);
    }

    @Override
    public double[] getDistances() {
        return distances;
    }

    @Override
    public void update(Instance ins) {
        instances.add(ins);
    }

    @Override
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 2 $");
    }
}
