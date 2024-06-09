//
//  UserResponse.swift
//  PinSample
//
//  Created by Ebraham Alskaf on 08/06/2024.
//  Copyright © 2024 Udacity. All rights reserved.
//

import Foundation

struct AuthenticationResponse: Codable {
    let user: User
}

struct User: Codable {
    let lastName: String
    let socialAccounts: [String]
    let mailingAddress: String?
    let cohortKeys: [String]
    let signature: String?
    let stripeCustomerID: String?
    let Guard: Guard
    let facebookID: String?
    let timezone: String?
    let sitePreferences: String?
    let occupation: String?
    let image: String?
    let firstName: String
    let jabberID: String?
    let languages: String?
    let badges: [String]
    let location: String?
    let externalServicePassword: String?
    let principals: [String]
    let enrollments: [String]
    let email: Email
    let websiteURL: String?
    let externalAccounts: [String]
    let bio: String?
    let coachingData: String?
    let tags: [String]
    let affiliateProfiles: [String]
    let hasPassword: Bool
    let emailPreferences: EmailPreferences
    let resume: String?
    let key: String
    let nickname: String
    let employerSharing: Bool
    let memberships: [Membership]
    let zendeskID: String?
    let registered: Bool
    let linkedinURL: String?
    let googleID: String?
    let imageURL: String
}

struct Guard: Codable {
    let canEdit: Bool
    let permissions: [Permission]
    let allowedBehaviors: [String]
    let subjectKind: String
}

struct Permission: Codable {
    let derivation: [String]
    let behavior: String
    let principalRef: PrincipalRef
}

struct PrincipalRef: Codable {
    let ref: String
    let key: String
}

struct Email: Codable {
    let verificationCodeSent: Bool
    let verified: Bool
    let address: String
}

struct EmailPreferences: Codable {
    let okUserResearch: Bool
    let masterOk: Bool
    let okCourse: Bool
}

struct Membership: Codable {
    let current: Bool
    let groupRef: PrincipalRef
    let creationTime: String?
    let expirationTime: String?
}
