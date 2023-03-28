show databases;
use zillow;
show tables;

-- bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips 
-- for all 'Single Family Residential' properties

describe properties_2016; 
select * from properties_2016 order by id;
select * from properties_2017;
select * from propertylandusetype;
describe properties_2017;
select count(*) from properties_2017;
select * from typeconstructiontype;


SELECT propertylandusetypeid
FROM propertylandusetype
WHERE propertylandusedesc = "Single Family Residential";

SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
FROM properties_2016
WHERE propertylandusetypeid IN(
	SELECT propertylandusetypeid
	FROM propertylandusetype
	WHERE propertylandusedesc = "Single Family Residential")
;

SELECT *
-- bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
-- taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid
FROM properties_2017
WHERE propertylandusetypeid IN(
	SELECT propertylandusetypeid
	FROM propertylandusetype
	WHERE propertylandusedesc = "Single Family Residential")
-- LIMIT 5
;