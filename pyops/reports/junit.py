import xml.etree.ElementTree as ET
from datetime import datetime

def to_junit_xml(results, suite_name="pyops"):
    testsuite = ET.Element("testsuite", {
        "name": suite_name,
        "timestamp": datetime.utcnow().isoformat(),
        "tests": str(len(results)),
        "failures": str(sum(1 for r in results if not r["ok"]))
    })
    for r in results:
        case = ET.SubElement(testsuite, "testcase", {
            "classname": r.get("dataset", "repo"),
            "name": r.get("check", "check")
        })
        if not r["ok"]:
            failure = ET.SubElement(case, "failure", {"message": r.get("message","")})
            failure.text = r.get("message","")
    return ET.tostring(testsuite, encoding="utf-8", xml_declaration=True)
